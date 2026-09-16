"""CPU regression cases for structured-value matching, never GPU timing."""

import unittest

from .implementation_match import Regions, compare
from .match_validation import reference_keys

NUM_GPUS = 0


def program(extra="", scale="v.x * 2.0f", input_name="x", bound="n"):
    return f"""__global__ void k(const float4* x, float4* y, int n) {{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i >= {bound}) return;
    float4 v={input_name}[i];
    v.x=v.x+1.0f;
    v.x=fmaxf(v.x,0.0f);
    v.x={scale};
    {extra}
    y[i]=v;
}}"""


def extract(source, context="same_context"):
    return Regions({"source": source, "line": 1, "templates": []}, context).result()


class MatchingChecks(unittest.TestCase):
    def test_identical(self):
        a = extract(program())
        self.assertEqual(compare(a, a)["kind"], "normalized_whole")

    def test_commutative_revision(self):
        self.assertEqual(compare(extract(program()), extract(program(scale="2.0f * v.x")))["kind"], "normalized_whole")

    def test_five_to_three(self):
        a = extract(program("v.x=v.x-3.0f; v.x=tanhf(v.x);"))
        b = extract(program())
        self.assertEqual(compare(a, b)["kind"], "partial_region")
        self.assertTrue(compare(a, b)["matched_regions"][0]["is_output_value"])

    def test_shared_prefix_not_whole_output_retention(self):
        a = extract(program())
        b = extract(program("v.x=v.x-3.0f; v.x=tanhf(v.x);"))
        self.assertEqual(compare(a, b)["kind"], "no_match")

    def test_wrong_scale(self):
        self.assertEqual(compare(extract(program()), extract(program(scale="v.x * 7.0f")))["kind"], "no_match")

    def test_wrong_input(self):
        self.assertEqual(compare(extract(program()), extract(program(input_name="y")))["kind"], "no_match")

    def test_wrong_output(self):
        self.assertEqual(
            compare(extract(program()), extract(program().replace("y[i]=v", "y[i+1]=v")))["kind"], "no_match"
        )

    def test_shared_scratch_not_output(self):
        s = (
            program()
            .replace("float4 v=x[i]", "__shared__ float4 scratch[32]; float4 v=x[i]")
            .replace("y[i]=v", "scratch[i]=v")
        )
        self.assertFalse(extract(s)["stages"])

    def test_guard_change(self):
        self.assertEqual(compare(extract(program()), extract(program(bound="n - 1")))["kind"], "no_match")

    def test_loop_field_update_is_not_ignored(self):
        changed = program("for(int j=0;j<4;j++){v.x+=j;}")
        self.assertEqual(compare(extract(program()), extract(changed))["kind"], "no_match")

    def test_standalone_increment_is_not_ignored(self):
        changed = extract(program("v.x++;"))
        self.assertEqual(compare(extract(program()), changed)["kind"], "no_match")
        self.assertIn("unsupported_expression_statement:update_expression", changed["unknowns"])

    def test_loop_increment_of_field_or_memory_is_unknown(self):
        for effect in ["v.x++", "y[i]++"]:
            with self.subTest(effect=effect):
                changed = extract(program(f"for(int j=0;j<4;j++){{{effect};}}"))
                self.assertEqual(compare(extract(program()), changed)["kind"], "no_match")
                self.assertIn("unresolved_loop_update", changed["unknowns"])

    def test_reference_exclusion_preserves_ast_identity(self):
        first = reference_keys("x = 1\n")
        reformatted = reference_keys("# comment\nx=1\n")
        changed = reference_keys("x = 2\n")
        self.assertEqual(len(first & reformatted), 1)
        self.assertFalse(first & changed)

    def test_unroll_factor_preserved(self):
        a = program("\n#pragma unroll 4\nfor(int j=0;j<4;j++){v.x+=j;}")
        b = a.replace("unroll 4", "unroll 8")
        self.assertEqual(compare(extract(a), extract(b))["kind"], "no_match")

    def test_macro_context_change(self):
        self.assertEqual(compare(extract(program(), "a"), extract(program(), "b"))["kind"], "no_match")

    def test_reassociation_not_normalized(self):
        a = extract(program(scale="(v.x + 2.0f) + 3.0f"))
        b = extract(program(scale="v.x + (2.0f + 3.0f)"))
        self.assertEqual(compare(a, b)["kind"], "no_match")

    def test_unknown_helper(self):
        a = extract(program(scale="mystery(v.x)"))
        self.assertIsNone(a["whole_normalized"])
        self.assertFalse(a["stages"])

    def test_shadowed_local_is_unknown(self):
        a = extract(program("{float v=0; y[i].x=v;}"))
        self.assertIn("repeated_local_declarations", a["unknowns"])
        self.assertFalse(a["stages"])

    def test_dead_prefix_not_reward_region(self):
        a = extract(program("v.x=99.0f;"))
        self.assertEqual(compare(extract(program("v.x=v.x-3.0f;")), a)["kind"], "no_match")

    def test_rename(self):
        a = program()
        b = a.replace("float4* x", "float4* inp").replace("=x[i]", "=inp[i]").replace("void k", "void another")
        self.assertEqual(compare(extract(a), extract(b))["kind"], "normalized_whole")


if __name__ == "__main__":
    unittest.main()
