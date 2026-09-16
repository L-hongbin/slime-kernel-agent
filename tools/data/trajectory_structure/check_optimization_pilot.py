"""Small CPU checks for the optimization-feature pilot; no generated CUDA execution."""

import unittest

from .optimization_pilot import inspect_kernel, load_timing, profile_matches


def inspect(source):
    return inspect_kernel({"source": source, "line": 1, "templates": []})


class OptimizationChecks(unittest.TestCase):
    def test_shared_staging(self):
        parsed = inspect(
            "__global__ void k(float* x){ __shared__ float a[32]; int i=threadIdx.x; a[i]=x[i]; __syncthreads(); x[i]=a[i]; }"
        )
        self.assertIn("shared_staging", parsed["features"])

    def test_shared_declaration_and_name_are_not_evidence(self):
        parsed = inspect("__global__ void fused_shared_kernel(float* x){ __shared__ float a[32]; x[0]=1; }")
        self.assertFalse(parsed["features"])

    def test_comments_are_not_evidence(self):
        parsed = inspect(
            "__global__ void k(float* x){ // __shfl_down_sync(1,x,1); __shared__ float a[32];\n x[0]=1; }"
        )
        self.assertFalse(parsed["features"])

    def test_shuffle(self):
        parsed = inspect("__global__ void k(float* x){ x[0]=__shfl_down_sync(0xffffffff,x[0],1); }")
        self.assertIn("warp_shuffle", parsed["features"])

    def test_vector_dereference(self):
        parsed = inspect("__global__ void k(float* x){ float4 v=reinterpret_cast<float4*>(x)[0]; x[0]=v.x; }")
        self.assertIn("vector_memory", parsed["features"])

    def test_vector_declaration_not_access(self):
        self.assertNotIn("vector_memory", inspect("__global__ void k(float* x){ float4 v; x[0]=1; }")["features"])

    def test_epilogue(self):
        source = (
            "__global__ void k(float* x, float* y, int n){ float s=0; for(int i=0;i<n;++i){s+=x[i];} y[0]=tanhf(s); }"
        )
        self.assertIn("reduction_epilogue", inspect(source)["features"])

    def test_raw_reduction_store_not_epilogue(self):
        source = "__global__ void k(float* x, float* y, int n){ float s=0; for(int i=0;i<n;++i){s+=x[i];} y[0]=s; }"
        self.assertNotIn("reduction_epilogue", inspect(source)["features"])

    def test_induction_variable_not_accumulator(self):
        source = "__global__ void k(float* x,int n){int i; for(i=0;i<n;i+=2){x[i]=1;} x[0]=i*2;}"
        self.assertNotIn("reduction_epilogue", inspect(source)["features"])

    def test_while_counter_not_accumulator(self):
        source = "__global__ void k(float* x,int n){int i=0; while(i<n){x[i]=1;i+=2;} x[0]=i*2;}"
        self.assertNotIn("reduction_epilogue", inspect(source)["features"])

    def test_unroll_request(self):
        source = "__global__ void k(float* x){\n#pragma unroll\nfor(int i=0;i<4;++i){x[i]=1;} }"
        self.assertIn("unroll_request", inspect(source)["features"])
        self.assertNotIn("unroll_request", inspect(source.replace("unroll", "unroll 1"))["features"])

    def test_renaming_and_arithmetic(self):
        a = inspect("__global__ void k(float* x){ int i=threadIdx.x; x[i]=x[i]+1; }")
        b = inspect("__global__ void other(float* q){ int j=threadIdx.x; q[j]=q[j]+1; }")
        c = inspect("__global__ void other(float* q){ int j=threadIdx.x; q[j]=q[j]*1; }")
        self.assertEqual(a["structure_hash"], b["structure_hash"])
        self.assertNotEqual(a["structure_hash"], c["structure_hash"])

    def test_profile_not_substring(self):
        profiles = [{"name": "not_k(float*)"}, {"name": "void k<128>(float*)"}]
        self.assertEqual(profile_matches("k", profiles), profiles[1:])

    def test_wrong_output_excluded(self):
        turn = {"complete_sections": True, "observation": {"correctness": False, "compiled": True, "speedup": 100}}
        self.assertIsNone(load_timing(turn, {})[0])

    def test_timing_ratio(self):
        turn = {
            "complete_sections": True,
            "observation": {"correctness": True, "compiled": True, "decoy": False, "speedup": 2},
        }
        raw = {"env_result": {"correctness": True, "compiled": True, "reference_runtime": 4, "kernel_runtime": 2}}
        self.assertIsNotNone(load_timing(turn, raw)[0])
        raw["env_result"]["kernel_runtime"] = 3
        self.assertEqual(load_timing(turn, raw)[1], "invalid_timing_ratio")


if __name__ == "__main__":
    unittest.main()
