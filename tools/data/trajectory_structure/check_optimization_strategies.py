"""CPU source examples for every registered strategy; no generated CUDA execution."""

import json
import tempfile
import unittest
from pathlib import Path

from .check_structure import response
from .optimization_coverage import run, scan_records, scan_snapshot, summarize
from .optimization_strategies import STRATEGIES, inspect_transition
from .structure import analyze_response


def kernel(body, params="float* x, float* y, int n", prefix=""):
    return f"{prefix}\n__global__ void k({params}) {{ {body} }}"


FIXTURES = [
    (
        {"warp_shuffle", "warp_collective"},
        kernel("x[0]=__shfl_down_sync(0xffffffff,x[0],1)+__reduce_add_sync(0xffffffff,1);"),
    ),
    (
        {"shared_staging", "shared_tiling", "shared_padding", "shared_transpose"},
        kernel(
            "__shared__ float s[32][33]; int tx=threadIdx.x, ty=threadIdx.y; for(int j=0;j<n;j++){ s[ty][tx]=x[j]; __syncthreads(); y[j]=s[tx][ty]; }"
        ),
    ),
    ({"vector_memory"}, kernel("float4 v=reinterpret_cast<float4*>(x)[0]; y[0]=v.x;")),
    ({"unroll_request"}, kernel("\n#pragma unroll 4\nfor(int i=0;i<4;i++){y[i]=x[i];}")),
    ({"reduction_epilogue"}, kernel("float s=0; for(int i=0;i<n;i++){s+=x[i];} y[0]=tanhf(s);")),
    ({"coalesced_access", "inplace_update"}, kernel("int i=blockIdx.x*blockDim.x+threadIdx.x; x[i]=x[i]*2;")),
    ({"read_only_cache"}, kernel("y[0]=__ldg(x);")),
    ({"cache_policy"}, kernel('asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(y[0]) : "l"(x));')),
    ({"register_tiling"}, kernel("float acc[4]={0}; for(int j=0;j<4;j++){acc[j]+=x[j];} y[0]=acc[0];")),
    ({"shared_swizzle"}, kernel("__shared__ float s[32][32]; int r=threadIdx.y,c=threadIdx.x; s[r][c^(r&7)]=x[0];")),
    ({"constant_memory"}, kernel("y[0]=coeff[0]*x[0];", prefix="__constant__ float coeff[16];")),
    (
        {"async_copy", "pipeline_staging"},
        kernel("cuda::memcpy_async(g,s,x,16,b); pipe.producer_commit(); pipe.consumer_wait();"),
    ),
    (
        {"async_copy", "tma_copy"},
        kernel(
            'asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global [%0], [%1], [%2];" :: "r"(s),"l"(x),"r"(b));'
        ),
    ),
    (
        {"double_buffer"},
        kernel(
            "__shared__ float s[2][256]; int stage=0,tid=threadIdx.x; for(int j=0;j<n;j++){s[stage&1][tid]=x[j];stage++;}"
        ),
    ),
    ({"prefetch"}, kernel('asm volatile("prefetch.global.L2 [%0];" :: "l"(x));')),
    ({"block_collective"}, kernel("y[0]=cub::BlockReduce<float,128>(temp).Sum(x[0]);")),
    ({"grid_stride"}, kernel("for(int i=threadIdx.x;i<n;i+=blockDim.x*gridDim.x){y[i]=x[i];}")),
    ({"thread_coarsening"}, kernel("int tid=threadIdx.x; for(int j=0;j<4;j++){y[tid+j*blockDim.x]=x[j];}")),
    ({"warp_specialization"}, kernel("int warp=threadIdx.x/32; if(warp==0){y[0]=x[0];}else{y[1]=x[1];}")),
    (
        {"persistent_queue", "atomic_accumulation"},
        kernel("while(n>0){int t=atomicAdd(q,1);y[t]=x[t];}", params="float* x, float* y, int* q, int n"),
    ),
    ({"tensor_core"}, kernel("nvcuda::wmma::mma_sync(d,a,b,c);")),
    ({"packed_math", "explicit_fma"}, kernel("y[0]=__hfma2(x[0],x[1],x[2]);", params="half2* x, half2* y, int n")),
    (
        {"mixed_precision_accumulation"},
        kernel(
            "float sum=0;for(int i=0;i<n;i++){sum+=__half2float(x[i]);}y[0]=sum;",
            params="const half* x, float* y, int n",
        ),
    ),
    ({"fast_math"}, kernel("y[0]=__expf(x[0]);")),
    ({"reciprocal_sqrt"}, kernel("y[0]=rsqrtf(x[0]);")),
    (
        {"restrict_aliasing", "dynamic_shared_memory", "warp_synchronization", "memory_fence"},
        kernel(
            "extern __shared__ float scratch[]; __syncwarp(); scratch[threadIdx.x]=x[threadIdx.x]; __threadfence_block(); y[threadIdx.x]=scratch[threadIdx.x];",
            params="const float* __restrict__ x, float* __restrict__ y, int n",
        ),
    ),
    (
        {"cooperative_grid_sync", "thread_block_cluster", "distributed_shared_memory"},
        kernel(
            "auto grid=cooperative_groups::this_grid(); auto cluster=cooperative_groups::this_cluster(); float* peer=cluster.map_shared_rank(s,0); grid.sync(); cluster.sync(); y[0]=peer[0];"
        ),
    ),
    ({"bitwise_indexing"}, kernel("int i=threadIdx.x; int row=i>>5,col=i&31;y[row*n+col]=x[i];")),
    ({"loop_invariant_hoisting"}, kernel("float inv=1.0f/n;for(int i=0;i<n;i++){y[i]=x[i]*inv;}")),
    (
        {"pointwise_fusion", "multi_output_fusion"},
        kernel("float v=x[0]+1; y[0]=fmaxf(v,0);z[0]=v;", params="float* x,float* y,float* z,int n"),
    ),
    ({"compile_time_specialization"}, "template<int N> __global__ void k(float*x){x[0]=N;}"),
    ({"launch_bounds"}, "__global__ __launch_bounds__(128,2) void k(float*x){x[0]=1;}"),
    (
        {"library_offload", "library_math_mode", "library_algorithm"},
        "void launch(){cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH);cublasGemmEx(h,a,b,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_EPILOGUE,e,4);}",
    ),
    ({"cuda_graph", "async_transfer"}, "void launch(){cudaGraphLaunch(g,s);cudaMemcpyAsync(a,b,16,kind,s);}"),
    (
        {"thread_block_cluster"},
        "void launch(){cudaLaunchConfig_t config{};config.attrs[0].id=cudaLaunchAttributeClusterDimension;cudaLaunchKernelEx(&config,k,a);}",
    ),
    (
        {
            "stream_event_scheduling",
            "cooperative_grid_sync",
            "thread_block_cluster",
            "pinned_host_memory",
            "managed_memory",
            "dynamic_shared_memory",
            "async_transfer",
        },
        "void launch(){cudaStreamCreate(&s);cudaEventRecord(e,s);cudaStreamWaitEvent(s,e,0);cudaLaunchCooperativeKernel(k,g,b,a,0,s);cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,1024);cudaFuncSetAttribute(k,cudaFuncAttributeNonPortableClusterSizeAllowed,1);cudaHostAlloc(&h,64,0);cudaMallocManaged(&m,64);cudaMemAdvise(m,64,cudaMemAdviseSetPreferredLocation,0);cudaMemPrefetchAsync(m,64,0,s);}",
    ),
]


class StrategyChecks(unittest.TestCase):
    def test_scalar_initialized_from_load_is_not_array_tile(self):
        observations, _, _ = scan_snapshot(
            analyze_response(response(kernel("float v=x[0];for(int j=0;j<n;j++){v+=x[j];}y[0]=v;")))
        )
        self.assertNotIn("register_tiling", {o["feature"] for o in observations})

    def test_index_arithmetic_is_not_pointwise_fusion(self):
        observations, _, _ = scan_snapshot(
            analyze_response(response(kernel("int i=blockIdx.x*blockDim.x+threadIdx.x;y[i]=x[i];")))
        )
        self.assertNotIn("pointwise_fusion", {o["feature"] for o in observations})

    def test_coverage_workflow_writes_complete_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "trajectory_0000.json"
            source.write_text(
                json.dumps(
                    [
                        dict(
                            group_id=0,
                            turn_idx=0,
                            response=response(kernel("y[0]=__expf(x[0]);")),
                            env_result={"correctness": False, "compiled": True},
                        )
                    ]
                )
            )
            output = Path(root) / "coverage"
            run([f"fixture={source}"], output)
            self.assertTrue(json.loads((output / "manifest.json").read_text())["code_and_inputs_unchanged"])
            self.assertEqual(json.loads((output / "summary.json").read_text())["trajectories"], 1)

    def test_every_snapshot_template_has_a_positive_example(self):
        covered = set()
        for expected, source in FIXTURES:
            with self.subTest(expected=sorted(expected)):
                snapshot = analyze_response(response(source))
                observations, _, rejected = scan_snapshot(snapshot)
                actual = {o["feature"] for o in observations}
                self.assertFalse(rejected, source)
                self.assertTrue(expected <= actual, f"missing={expected-actual}; actual={actual}; source={source}")
                covered.update(expected)
        self.assertEqual(covered, {k for k, r in STRATEGIES.items() if r["scope"] != "transition"})

    def test_fusion_fission_and_removed_buffer(self):
        a = analyze_response(
            response("void launch(float*x,float*y){auto tmp=empty(n); A<<<1,32>>>(x,tmp);B<<<1,32>>>(tmp,y);}")
        )
        b = analyze_response(response("void launch(float*x,float*y){F<<<1,32>>>(x,y);}"))
        self.assertEqual(
            {o["feature"] for o in inspect_transition(a, b)}, {"launch_fusion", "intermediate_elimination"}
        )
        self.assertEqual({o["feature"] for o in inspect_transition(b, a)}, {"launch_fission"})

    def test_launch_configuration(self):
        a = analyze_response(response("void launch(float*x){k<4><<<1,32>>>(x);}"))
        b = analyze_response(response("void launch(float*x){k<8><<<1,64>>>(x);}"))
        self.assertEqual({o["feature"] for o in inspect_transition(a, b)}, {"launch_config_change"})

    def test_failure_without_profiling_is_analyzed(self):
        raw = dict(
            group_id=0,
            turn_idx=0,
            response=response(kernel("y[0]=__expf(x[0]);")),
            env_result={"correctness": False, "compiled": True, "error_message": "runtime failure"},
        )
        result = scan_records([raw], ["example", "L3", "ref", "0"])
        self.assertIn("fast_math", {o["feature"] for o in result["turns"][0]["observations"]})
        self.assertEqual(result["turns"][0]["evaluation_state"], "observed_failure")
        self.assertIsNone(result["turns"][0]["timing"])
        self.assertEqual(summarize([result])["failed_source_strategy"]["turns"], 1)

    def test_plain_copy_does_not_imply_explicit_strategy(self):
        snapshot = analyze_response(response(kernel("int i=threadIdx.x;y[i]=x[i];")))
        observations, _, _ = scan_snapshot(snapshot)
        self.assertTrue(observations)
        self.assertFalse(any(STRATEGIES[o["feature"]]["evidence"] == "explicit" for o in observations))

    def test_rsqrtf_is_not_generic_fast_math(self):
        observations, _, _ = scan_snapshot(analyze_response(response(kernel("y[0]=rsqrtf(x[0]);"))))
        features = {o["feature"] for o in observations}
        self.assertIn("reciprocal_sqrt", features)
        self.assertNotIn("fast_math", features)

    def test_comments_and_strings_do_not_invoke_apis(self):
        observations, _, _ = scan_snapshot(
            analyze_response(response(kernel('// __expf(x); cp.async.bulk;\nconst char* p="mma.sync"; y[0]=1;')))
        )
        self.assertFalse({o["feature"] for o in observations} & {"fast_math", "tensor_core", "async_copy", "tma_copy"})

    def test_strided_access_not_reported_lane_contiguous(self):
        observations, _, _ = scan_snapshot(analyze_response(response(kernel("int i=threadIdx.x*32; y[i]=x[i];"))))
        self.assertNotIn("coalesced_access", {o["feature"] for o in observations})

    def test_single_stage_is_not_double_buffer(self):
        observations, _, _ = scan_snapshot(
            analyze_response(response(kernel("__shared__ float s[4][32];s[threadIdx.y][threadIdx.x]=1;")))
        )
        self.assertNotIn("double_buffer", {o["feature"] for o in observations})


if __name__ == "__main__":
    unittest.main()
