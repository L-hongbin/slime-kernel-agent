"""Task-independent optimization vocabulary and source evidence.

One pass collects observations; no retrieval/reranking filter. Templates describe
source mechanisms or cross-turn changes, not measured performance benefits.
"""

from __future__ import annotations

import collections
import re

from .structure import PARSERS, callable_name, native_tokens, text, walk


def rule(family, title, template, example, scope="kernel", evidence="structure"):
    return dict(family=family, title=title, template=template, example=example, scope=scope, evidence=evidence)


# Canonical inventory, also dumped into every coverage run.
STRATEGIES = {
    "warp_shuffle": rule(
        "parallel", "Warp shuffle", "Actual __shfl* call", "__shfl_down_sync(mask, v, 16)", evidence="explicit"
    ),
    "shared_staging": rule(
        "memory",
        "Shared staging",
        "Same shared array read/written plus block barrier",
        "s[t]=x[t]; __syncthreads(); y[t]=s[t]",
    ),
    "vector_memory": rule(
        "memory",
        "Vector memory",
        "Indexed/dereferenced explicit vector pointer",
        "reinterpret_cast<float4*>(x)[i]",
        evidence="explicit",
    ),
    "unroll_request": rule(
        "compute",
        "Loop unrolling",
        "Pragma unroll immediately before for, excluding 0/1",
        "#pragma unroll 4",
        evidence="explicit",
    ),
    "reduction_epilogue": rule(
        "fusion",
        "Reduction epilogue",
        "Loop-carried scalar used in transformed output store",
        "for(...) sum+=x[i]; y[0]=tanhf(sum)",
    ),
    "coalesced_access": rule(
        "memory",
        "Lane-contiguous access",
        "Global subscript affine in threadIdx.x with coefficient +1/-1",
        "int i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i]",
    ),
    "read_only_cache": rule("memory", "Read-only cache load", "Actual __ldg call", "__ldg(x+i)", evidence="explicit"),
    "restrict_aliasing": rule(
        "memory",
        "Restrict-qualified pointer interface",
        "Pointer kernel parameter explicitly carries __restrict__",
        "const float* __restrict__ x",
        evidence="explicit",
    ),
    "dynamic_shared_memory": rule(
        "memory",
        "Dynamic shared memory",
        "extern __shared__ array declaration or dynamic-shared function attribute",
        "extern __shared__ float scratch[]",
        evidence="explicit",
    ),
    "cache_policy": rule(
        "memory",
        "Explicit cache policy",
        "Cache-modified load/store intrinsic, PTX or runtime cache-policy API",
        'asm("ld.global.cg.f32 ...")',
        scope="native",
        evidence="explicit",
    ),
    "shared_tiling": rule(
        "reuse",
        "Shared-memory tiling",
        "Multidimensional shared array accessed in a loop with shared staging",
        "__shared__ float tile[16][16]; for(...) { tile[r][c]=x[i]; ... }",
    ),
    "register_tiling": rule(
        "reuse",
        "Thread-local accumulator tile",
        "Nonshared local array updated in a loop",
        "float acc[4]; for(...) acc[j]+=x[i]",
    ),
    "shared_padding": rule(
        "memory",
        "Shared bank padding",
        "Multidimensional shared array has a padded final extent",
        "__shared__ float tile[32][33]",
    ),
    "shared_swizzle": rule(
        "memory",
        "Shared index swizzle",
        "XOR in shared indexing, including local index definitions",
        "tile[row][col ^ (row & 7)]",
    ),
    "shared_transpose": rule(
        "memory",
        "Shared layout transpose",
        "Same shared array read/written with reversed 2D subscripts",
        "tile[ty][tx]=x[i]; y[j]=tile[tx][ty]",
    ),
    "constant_memory": rule(
        "memory",
        "Constant-memory access",
        "Kernel references a symbol declared __constant__",
        "__constant__ float coeff[16]; y[i]=coeff[k]*x[i]",
        scope="context",
        evidence="explicit",
    ),
    "async_copy": rule(
        "pipeline",
        "Asynchronous device copy",
        "memcpy_async / pipeline copy API or cp.async PTX",
        "cuda::memcpy_async(group, shared, global, bytes, barrier)",
        evidence="explicit",
    ),
    "tma_copy": rule(
        "pipeline",
        "Bulk tensor copy / TMA",
        "cp.async.bulk(.tensor) PTX or matching cuda::ptx intrinsic",
        'asm("cp.async.bulk.tensor.2d.shared::cluster.global ...")',
        evidence="explicit",
    ),
    "pipeline_staging": rule(
        "pipeline",
        "Copy/compute pipeline",
        "Producer/consumer pipeline or async commit/wait operations",
        "pipe.producer_commit(); pipe.consumer_wait()",
        evidence="explicit",
    ),
    "double_buffer": rule(
        "pipeline",
        "Double buffering",
        "Two-slot shared array indexed with alternating &1 or %2 stage",
        "__shared__ float s[2][256]; s[stage & 1][tid]=x[i]",
    ),
    "prefetch": rule(
        "pipeline",
        "Explicit prefetch",
        "PTX prefetch instruction or named CUDA PTX prefetch intrinsic",
        'asm("prefetch.global.L2 [%0];" :: "l"(x))',
        evidence="explicit",
    ),
    "warp_collective": rule(
        "parallel",
        "Warp collectives",
        "Warp reduce/vote/match intrinsic or cooperative-group collective",
        "__reduce_add_sync(mask, v)",
        evidence="explicit",
    ),
    "block_collective": rule(
        "parallel",
        "Block/warp collective library",
        "CUB Block/ Warp Reduce/Scan/Load/Store type and collective member call",
        "cub::BlockReduce<float,128>(temp).Sum(v)",
        evidence="explicit",
    ),
    "grid_stride": rule(
        "parallel",
        "Grid-stride loop",
        "Loop update depends on blockDim and gridDim",
        "for(int i=tid;i<n;i+=blockDim.x*gridDim.x) y[i]=x[i]",
    ),
    "thread_coarsening": rule(
        "parallel",
        "Multiple outputs per thread",
        "Loop writes global indices depending on loop counter and thread index",
        "for(int j=0;j<4;j++) y[tid+j*blockDim.x]=v",
    ),
    "warp_specialization": rule(
        "parallel",
        "Warp-role partitioning",
        "Conditional depends on threadIdx divided by 32 or shifted by 5",
        "int warp=threadIdx.x/32; if(warp==0) { ... } else { ... }",
    ),
    "persistent_queue": rule(
        "parallel",
        "Persistent work queue",
        "While/do loop obtains work using atomic increment",
        "while(...) { task=atomicAdd(queue,1); ... }",
    ),
    "atomic_accumulation": rule(
        "parallel",
        "Atomic partial-result accumulation",
        "Atomic add/max/min/reduction invocation",
        "atomicAdd(out+index, partial)",
        evidence="explicit",
    ),
    "tensor_core": rule(
        "compute",
        "Tensor Core MMA",
        "WMMA/MMA intrinsic or mma/wgmma/tcgen05.mma PTX",
        "nvcuda::wmma::mma_sync(d,a,b,c)",
        evidence="explicit",
    ),
    "packed_math": rule(
        "compute",
        "Packed low-precision arithmetic",
        "Actual half2/bfloat162 arithmetic intrinsic",
        "__hfma2(a,b,c)",
        evidence="explicit",
    ),
    "mixed_precision_accumulation": rule(
        "compute",
        "Low-precision input with FP32 accumulation",
        "Half/bfloat input, float accumulator and scalar self-update",
        "const half* x; float sum=0; for(...) sum+=__half2float(x[i])",
    ),
    "fast_math": rule(
        "compute",
        "Approximate/fast math",
        "Fast math intrinsic or approx PTX operation",
        "__expf(x) + __fdividef(a,b)",
        evidence="explicit",
    ),
    "reciprocal_sqrt": rule(
        "compute",
        "Reciprocal square-root API",
        "Actual rsqrtf call or rsqrt PTX operation; no approximate claim for the C API",
        "rsqrtf(x)",
        evidence="explicit",
    ),
    "explicit_fma": rule(
        "compute",
        "Explicit fused multiply-add",
        "fma/fmaf or packed fma call / PTX fma",
        "fmaf(a,b,c)",
        evidence="explicit",
    ),
    "bitwise_indexing": rule(
        "compute",
        "Bitwise index arithmetic",
        "Shift/mask in an array index or its local definitions",
        "int row=i>>5; int col=i&31; y[row*stride+col]=v",
    ),
    "loop_invariant_hoisting": rule(
        "compute",
        "Precomputed loop input",
        "Arithmetic/call initializer before loop used inside loop, not assigned in loop",
        "float inv=1.0f/n; for(...) y[i]=x[i]*inv",
    ),
    "pointwise_fusion": rule(
        "fusion",
        "Chained pointwise computation",
        "Output expression and local producers contain at least two compute operations",
        "float v=x[i]+bias; y[i]=fmaxf(v,0)",
    ),
    "multi_output_fusion": rule(
        "fusion",
        "Multiple-output kernel",
        "Kernel stores to at least two distinct pointer parameters",
        "y[i]=f(x[i]); z[i]=g(x[i])",
    ),
    "inplace_update": rule(
        "reuse", "In-place update", "Same global pointer parameter read and written", "x[i]=x[i]*scale"
    ),
    "compile_time_specialization": rule(
        "scheduling",
        "Compile-time specialization",
        "Template kernel, constexpr declaration or if constexpr",
        "template<int N> __global__ void k(...) { ... }",
        evidence="explicit",
    ),
    "launch_bounds": rule(
        "scheduling",
        "Resource/occupancy configuration",
        "__launch_bounds__ or CUDA function attribute/cache/occupancy API",
        "__launch_bounds__(128,2)",
        scope="native",
        evidence="explicit",
    ),
    "library_offload": rule(
        "library",
        "Optimized library computation",
        "Actual cuBLAS/cuDNN compute call or CUTLASS GEMM invocation",
        "cublasGemmEx(handle, ...)",
        scope="native",
        evidence="explicit",
    ),
    "library_math_mode": rule(
        "library",
        "Library precision/math mode",
        "Library call carrying math/compute mode or setting a mode",
        "cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH)",
        scope="native",
        evidence="explicit",
    ),
    "library_algorithm": rule(
        "library",
        "Library algorithm/epilogue configuration",
        "Algorithm selection API, explicit algorithm or matmul epilogue attribute",
        "cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,...)",
        scope="native",
        evidence="explicit",
    ),
    "cuda_graph": rule(
        "scheduling",
        "CUDA Graph execution",
        "CUDA graph capture/instantiate/launch API",
        "cudaGraphLaunch(graph,stream)",
        scope="native",
        evidence="explicit",
    ),
    "stream_event_scheduling": rule(
        "scheduling",
        "Stream/event scheduling",
        "CUDA stream/event creation, record, wait, synchronization or destruction API",
        "cudaEventRecord(done, stream); cudaStreamWaitEvent(other, done, 0)",
        scope="native",
        evidence="explicit",
    ),
    "cooperative_grid_sync": rule(
        "synchronization",
        "Cooperative grid synchronization",
        "Cooperative launch API or this_grid/grid sync call",
        "cudaLaunchCooperativeKernel(...); grid.sync()",
        scope="native",
        evidence="explicit",
    ),
    "warp_synchronization": rule(
        "synchronization", "Warp synchronization", "Actual __syncwarp call", "__syncwarp(mask)", evidence="explicit"
    ),
    "memory_fence": rule(
        "synchronization",
        "Memory fence",
        "Actual __threadfence* call or membar PTX instruction",
        "__threadfence_system()",
        evidence="explicit",
    ),
    "thread_block_cluster": rule(
        "scheduling",
        "Thread-block cluster",
        "Cluster launch/configuration API or cooperative-groups cluster synchronization",
        "cudaLaunchAttributeClusterDimension; cluster.sync()",
        scope="native",
        evidence="explicit",
    ),
    "distributed_shared_memory": rule(
        "memory",
        "Distributed shared memory",
        "Cluster shared-memory mapping API or shared::cluster PTX",
        "cluster.map_shared_rank(smem, rank)",
        scope="native",
        evidence="explicit",
    ),
    "pinned_host_memory": rule(
        "memory",
        "Pinned host memory",
        "cudaHostAlloc/cudaMallocHost/cudaHostRegister API",
        "cudaHostAlloc(&host, bytes, cudaHostAllocDefault)",
        scope="native",
        evidence="explicit",
    ),
    "managed_memory": rule(
        "memory",
        "Managed-memory placement",
        "cudaMallocManaged, cudaMemAdvise or cudaMemPrefetchAsync API",
        "cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device)",
        scope="native",
        evidence="explicit",
    ),
    "async_transfer": rule(
        "pipeline",
        "Asynchronous host/device transfer",
        "cudaMemcpy*Async or cudaMemPrefetchAsync invocation",
        "cudaMemcpyAsync(dst,src,bytes,kind,stream)",
        scope="native",
        evidence="explicit",
    ),
    "launch_fusion": rule(
        "fusion",
        "Launch fusion candidate",
        "Matched host function changes multiple launch sites to fewer new sites",
        "A(...); B(...); C(...) -> F(...)",
        scope="transition",
    ),
    "launch_fission": rule(
        "fusion",
        "Launch splitting candidate",
        "Matched host function changes launch sites to more new sites",
        "F(...) -> ABC(...); D(...); E(...)",
        scope="transition",
    ),
    "launch_config_change": rule(
        "scheduling",
        "Launch configuration change",
        "Same kernel call site changes launch configuration or template arguments",
        "k<<<grid,128>>>(x) -> k<<<grid,256>>>(x)",
        scope="transition",
    ),
    "intermediate_elimination": rule(
        "reuse",
        "Intermediate-buffer removal candidate",
        "Removed allocation was shared by launch arguments before launch fusion",
        "tmp=empty(...); A(x,tmp); B(tmp,y) -> F(x,y)",
        scope="transition",
    ),
}

BASELINE_RULES = frozenset({"warp_shuffle", "shared_staging", "vector_memory", "unroll_request", "reduction_epilogue"})
LOOPS = {"for_statement", "while_statement", "do_statement"}


def compact(value):
    return re.sub(r"\s+", "", value)


def array_parts(node):
    """Base and ordered dimensions of a nested subscript expression."""
    dims = []
    while node is not None and node.type == "subscript_expression":
        index = node.child_by_field_name("indices") or node.child_by_field_name("index")
        values = index.named_children if index is not None and index.type == "subscript_argument_list" else [index]
        dims[0:0] = [text(v) for v in values if v is not None]
        node = node.child_by_field_name("argument")
    return text(node), dims


def access_mode(node):
    parent = node.parent
    while parent is not None and parent.type in {
        "field_expression",
        "parenthesized_expression",
        "subscript_expression",
    }:
        node, parent = parent, parent.parent
    if parent is not None and parent.type == "assignment_expression":
        left = parent.child_by_field_name("left")
        if left is not None and left.start_byte <= node.start_byte and node.end_byte <= left.end_byte:
            return {"write"} if text(parent.child_by_field_name("operator")) == "=" else {"read", "write"}
    return {"read"}


def declarations(fn):
    result = {}
    for n in walk(fn):
        if n.type not in {"declaration", "parameter_declaration"}:
            continue
        for decl in n.children_by_field_name("declarator"):
            name = callable_name(decl)
            if name is not None:
                storage = decl.child_by_field_name("declarator") if decl.type == "init_declarator" else decl
                result[text(name)] = dict(
                    node=n,
                    declarator=decl,
                    init=decl.child_by_field_name("value"),
                    shared="__shared__" in native_tokens(n),
                    parameter=n.type == "parameter_declaration",
                    array_dims=re.findall(r"\[([^\]]+)\]", text(storage)),
                )
    return result


def expanded(value, decls, seen=frozenset()):
    """Bounded initializer substitution for source index/loop evidence."""
    if len(seen) >= 8:
        return value

    def replace(match):
        name = match[0]
        d = decls.get(name)
        if d and d["init"] is not None and name not in seen:
            return "(" + expanded(text(d["init"]), decls, seen | {name}) + ")"
        return name

    return re.sub(r"\b[A-Za-z_]\w*\b", replace, value)


def compute_operations(node, decls, seen=frozenset()):
    """Count value computations, treating memory addresses as load leaves."""
    if node is None or len(seen) >= 8:
        return 0
    if node.type in {"subscript_expression", "pointer_expression"}:
        return 0
    if node.type == "identifier":
        name = text(node)
        d = decls.get(name)
        return compute_operations(d["init"], decls, seen | {name}) if d and name not in seen else 0
    count = int(node.type in {"binary_expression", "conditional_expression"})
    if node.type == "call_expression":
        target = text(node.child_by_field_name("function"))
        if target in {"__ldg", "__ldca", "__ldcg", "__ldcs"}:
            return 0
        count = int(not target.startswith(("reinterpret_cast", "static_cast", "const_cast")))
        args = node.child_by_field_name("arguments")
        return count + sum(compute_operations(n, decls, seen) for n in args.named_children) if args else count
    return count + sum(compute_operations(n, decls, seen) for n in node.named_children)


def lane_coefficient(node, decls, seen=frozenset()):
    """Limited affine coefficient of threadIdx.x; nonlinear forms stay unknown."""
    if node is None:
        return None
    code = compact(text(node))
    if code == "threadIdx.x":
        return 1
    if node.type == "identifier":
        d = decls.get(code)
        if d and d["init"] is not None and code not in seen and len(seen) < 8:
            return lane_coefficient(d["init"], decls, seen | {code})
        return 0
    if node.type == "parenthesized_expression" and node.named_children:
        return lane_coefficient(node.named_children[0], decls, seen)
    if node.type == "binary_expression":
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        a, b = lane_coefficient(left, decls, seen), lane_coefficient(right, decls, seen)
        if a is None or b is None:
            return None
        op = text(node.child_by_field_name("operator"))
        if op in {"+", "-"}:
            return a + (b if op == "+" else -b)
        if a == b == 0:
            return 0
        if op == "*":
            if a == 0 and re.fullmatch(r"\d+", text(left)):
                return int(text(left)) * b
            if b == 0 and re.fullmatch(r"\d+", text(right)):
                return int(text(right)) * a
        return None
    return None if "threadIdx" in code else 0


def extend_kernel_features(component, fn, clean, evidence):
    body = fn.child_by_field_name("body")
    nodes = list(walk(body))
    decls = declarations(fn)
    shared = {k for k, d in decls.items() if d["shared"]}
    pointers = {
        k
        for k, d in decls.items()
        if d["parameter"] and ("*" in text(d["declarator"]) or "[" in text(d["declarator"]))
    }
    calls = [n for n in nodes if n.type == "call_expression"]
    loops = [n for n in nodes if n.type in LOOPS]
    accesses = [
        n
        for n in nodes
        if n.type == "subscript_expression" and not (n.parent and n.parent.type == "subscript_expression")
    ]

    def add(label, node, reason, **parameters):
        evidence[label].append(
            dict(
                line=component["line"] + node.start_point.row,
                code=text(node)[:900],
                reason=reason,
                parameters=parameters,
            )
        )

    for n in calls:
        target = text(n.child_by_field_name("function"))
        checks = {
            "read_only_cache": target == "__ldg",
            "warp_collective": bool(
                re.search(
                    r"(?:__reduce_\w+_sync|__ballot_sync|__all_sync|__any_sync|__match_\w+_sync|(?:cooperative_groups|cg)::(?:reduce|inclusive_scan|exclusive_scan))$",
                    target,
                )
            ),
            "atomic_accumulation": bool(re.fullmatch(r"atomic(?:Add|Max|Min|And|Or|Xor)(?:_block|_system)?", target)),
            "async_copy": bool(re.search(r"(?:memcpy_async|__pipeline_memcpy_async|cp_async(?:_\w+)?)$", target)),
            "tma_copy": "cp_async_bulk" in target,
            "pipeline_staging": bool(
                re.search(
                    r"(?:producer_acquire|producer_commit|consumer_wait|consumer_release|__pipeline_commit|__pipeline_wait_prior|cp_async_commit_group|cp_async_wait_group)$",
                    target,
                )
            ),
            "prefetch": bool(re.search(r"^(?:cuda::ptx::)prefetch(?:_\w+)?$", target)),
            "tensor_core": bool(re.search(r"(?:wmma::mma_sync|mma_sync|mma_async|wgmma|tcgen05_mma)", target)),
            "packed_math": bool(re.fullmatch(r"__(?:hadd2|hsub2|hmul2|h2div|hfma2|hmax2|hmin2)(?:_rn|_sat)?", target)),
            "fast_math": bool(
                re.fullmatch(r"__(?:expf|exp2f|logf|log2f|log10f|sinf|cosf|tanf|sincosf|powf|fdividef)", target)
            ),
            "reciprocal_sqrt": target == "rsqrtf",
            "explicit_fma": bool(re.fullmatch(r"fma[f]?|__fma[fd]_\w+|__hfma2(?:_\w+)?", target)),
            "warp_synchronization": target == "__syncwarp",
            "memory_fence": bool(re.fullmatch(r"__threadfence(?:_block|_system)?", target)),
            "cooperative_grid_sync": bool(re.fullmatch(r"(?:grid|this_grid\(\))\.sync", target)),
            "thread_block_cluster": bool(re.fullmatch(r"(?:cluster|this_cluster\(\))\.sync", target)),
            "distributed_shared_memory": bool(re.fullmatch(r"(?:cluster|this_cluster\(\))\.map_shared_rank", target)),
        }
        for label, matched in checks.items():
            if matched:
                add(label, n, "explicit API invocation", target=target)
    asm = [n for n in nodes if "asm" in n.type and not (n.parent and "asm" in n.parent.type)]
    patterns = {
        "async_copy": r"\bcp\.async\.",
        "tma_copy": r"\bcp\.async\.bulk",
        "pipeline_staging": r"\b(?:cp\.async\.(?:commit_group|wait_group|wait_all)|mbarrier\.)",
        "prefetch": r"\bprefetch(?:u)?\.",
        "tensor_core": r"\b(?:mma\.|wgmma\.mma|tcgen05\.mma)",
        "fast_math": r"\b(?:ex2|lg2|rcp|sin|cos)\.approx",
        "reciprocal_sqrt": r"\brsqrt(?:\.\w+)?",
        "explicit_fma": r"\bfma\.",
        "cache_policy": r"\b(?:ld|st)\.global\.(?:ca|cg|cs|cv|lu|wb|wt|nc)\b",
        "memory_fence": r"\bmembar(?:\.\w+)?",
        "thread_block_cluster": r"\b(?:clusterlaunchcontrol|barrier\.cluster|mbarrier\.cluster)",
        "distributed_shared_memory": r"\b(?:shared::cluster|mapa\.shared::cluster)",
    }
    for n in asm:
        for label, pattern in patterns.items():
            if re.search(pattern, text(n)):
                add(label, n, "inline PTX instruction", pattern=pattern)
    if re.search(r"\bcub::(?:Block|Warp)(?:Reduce|Scan|Load|Store)\s*<", clean) and any(
        re.search(
            r"[.]\s*(?:Sum|Reduce|InclusiveSum|ExclusiveSum|Load|Store)\s*$", text(n.child_by_field_name("function"))
        )
        for n in calls
    ):
        add("block_collective", body, "CUB collective type and invocation")
    modes = collections.defaultdict(set)
    shared_indices = collections.defaultdict(lambda: collections.defaultdict(list))
    for n in accesses:
        base, indices = array_parts(n)
        access = access_mode(n)
        modes[base].update(access)
        expanded_index = expanded(" ".join(indices), decls)
        if base in shared:
            for mode in access:
                shared_indices[base][mode].append(indices)
            if "^" in expanded_index:
                add("shared_swizzle", n, "shared index uses XOR", array=base, indices=indices)
            dims = decls[base]["array_dims"]
            if dims and compact(dims[0]) == "2" and re.search(r"&\s*1\b|%\s*2\b", expanded_index):
                add("double_buffer", n, "alternating access to two-slot shared storage", array=base)
        if base in pointers:
            index = n.child_by_field_name("indices") or n.child_by_field_name("index")
            idx = (
                index.named_children[0]
                if index is not None and index.type == "subscript_argument_list" and index.named_children
                else index
            )
            if lane_coefficient(idx, decls) in {1, -1}:
                add("coalesced_access", n, "affine lane-contiguous source index", index=indices)
            if re.search(r">>|<<|(?<!&)&(?!&)|\^", expanded_index):
                add("bitwise_indexing", n, "bitwise arithmetic in index or its initializers", index=indices)
    for name in shared:
        d = decls[name]
        dims = d["array_dims"]
        if "extern" in native_tokens(d["node"]):
            add("dynamic_shared_memory", d["node"], "extern shared array declaration", array=name)
        if modes[name] == {"read", "write"} and any(
            text(n.child_by_field_name("function")) == "__syncthreads" for n in calls
        ):
            if not any(e.get("parameters", {}).get("array") == name for e in evidence["shared_staging"]):
                add("shared_staging", d["node"], "shared array read/written with block barrier", array=name)
        if len(dims) >= 2:
            last = compact(dims[-1])
            if re.search(r"\+1$", last) or (
                last.isdigit() and int(last) > 2 and ((int(last) - 1) & (int(last) - 2)) == 0
            ):
                add("shared_padding", d["node"], "padded shared row extent", extents=dims)
            if (
                modes[name] == {"read", "write"}
                and evidence.get("shared_staging")
                and any(name in native_tokens(loop) for loop in loops)
            ):
                add("shared_tiling", d["node"], "multidimensional staged shared tile used in loop", extents=dims)
        reads, writes = shared_indices[name]["read"], shared_indices[name]["write"]
        if any(len(a) == len(b) == 2 and a == list(reversed(b)) and a[0] != a[1] for a in reads for b in writes):
            add("shared_transpose", d["node"], "shared write/read transpose", array=name)
    for n in nodes:
        if n.type == "if_statement":
            condition = expanded(text(n.child_by_field_name("condition")), decls)
            if re.search(r"threadIdx\s*\.\s*x\s*\)*\s*(?:/\s*32|>>\s*5)", condition):
                add("warp_specialization", n, "warp-id-based branch", condition=condition)
    for loop in loops:
        update = expanded(text(loop.child_by_field_name("update")), decls)
        if "blockDim" in update and "gridDim" in update:
            add("grid_stride", loop, "loop advances by grid-wide thread count", update=update)
        loop_body = loop.child_by_field_name("body")
        if loop_body is None:
            continue
        local_nodes = list(walk(loop_body))
        if loop.type in {"while_statement", "do_statement"} and any(
            n.type == "call_expression" and text(n.child_by_field_name("function")) in {"atomicAdd", "atomicInc"}
            for n in local_nodes
        ):
            add("persistent_queue", loop, "repeated atomic work acquisition")
        for n in local_nodes:
            if n.type != "assignment_expression":
                continue
            base, indices = array_parts(n.child_by_field_name("left"))
            d = decls.get(base)
            if (
                d
                and not d["shared"]
                and not d["parameter"]
                and d["array_dims"]
                and text(n.child_by_field_name("operator")) != "="
            ):
                add("register_tiling", n, "thread-local array accumulation", array=base)
            if base in pointers and "threadIdx" in expanded(" ".join(indices), decls):
                init = loop.child_by_field_name("initializer")
                loop_names = (
                    {
                        text(callable_name(d))
                        for node in walk(init)
                        if node.type == "declaration"
                        for d in node.children_by_field_name("declarator")
                    }
                    if init
                    else set()
                )
                if loop_names.intersection(re.findall(r"\b\w+\b", " ".join(indices))):
                    add("thread_coarsening", n, "loop iterates per-thread output positions", indices=indices)
        loop_tokens = set(native_tokens(loop_body))
        assigned = {text(n.child_by_field_name("left")) for n in local_nodes if n.type == "assignment_expression"}
        assigned.update(text(n.child_by_field_name("argument")) for n in local_nodes if n.type == "update_expression")
        for name, d in decls.items():
            init = d["init"]
            if (
                init is not None
                and d["node"].end_byte <= loop.start_byte
                and name in loop_tokens
                and name not in assigned
                and any(n.type in {"binary_expression", "call_expression"} for n in walk(init))
            ):
                add("loop_invariant_hoisting", d["node"], "computed value reused inside later loop", variable=name)
    low_input = any(
        d["parameter"]
        and re.search(r"\b(?:half|__half|half2|__nv_bfloat16|nv_bfloat16|__nv_bfloat162)\b", text(d["node"]))
        for d in decls.values()
    )
    if low_input:
        floats = {
            name
            for name, d in decls.items()
            if not d["parameter"] and text(d["node"].child_by_field_name("type")) == "float"
        }
        for n in nodes:
            if (
                n.type == "assignment_expression"
                and text(n.child_by_field_name("left")) in floats
                and text(n.child_by_field_name("operator")) in {"+=", "-=", "*=", "/="}
            ):
                add("mixed_precision_accumulation", n, "FP32 scalar accumulation with low-precision input parameter")
    stores = [
        n
        for n in nodes
        if n.type == "assignment_expression" and array_parts(n.child_by_field_name("left"))[0] in pointers
    ]
    outputs = {array_parts(n.child_by_field_name("left"))[0] for n in stores}
    if len(outputs) >= 2:
        add("multi_output_fusion", body, "multiple pointer outputs in one kernel", outputs=sorted(outputs))
    for name in pointers:
        if modes[name] == {"read", "write"}:
            add("inplace_update", body, "global buffer read and written by same kernel", buffer=name)
    for name, d in decls.items():
        if d["parameter"] and name in pointers and "__restrict__" in native_tokens(d["node"]):
            add("restrict_aliasing", d["node"], "restrict-qualified pointer parameter", parameter=name)
    for store in stores:
        expression = expanded(text(store.child_by_field_name("right")), decls)
        if compute_operations(store.child_by_field_name("right"), decls) >= 2:
            add(
                "pointwise_fusion",
                store,
                "chained scalar computation before output store",
                expression=expression[:900],
            )
    if component.get("templates") or re.search(r"\bconstexpr\b", clean):
        add(
            "compile_time_specialization",
            fn,
            "compile-time parameter or constexpr expression",
            templates=component.get("templates", []),
        )
    if "__launch_bounds__" in clean:
        add("launch_bounds", fn, "explicit launch-bounds annotation")


def inspect_program(snapshot):
    """Native host/configuration and device file-context observations."""
    result = []
    constants = set()
    for section in ("CUDA_KERNELS", "APPLY_BINDINGS"):
        source = snapshot["sections"].get(section, {}).get("source", "")
        root = PARSERS[section].parse(source.encode()).root_node
        for n in walk(root):
            if n.type == "declaration" and "__constant__" in native_tokens(n):
                constants.update(
                    text(callable_name(d)) for d in n.children_by_field_name("declarator") if callable_name(d)
                )
    for c in snapshot["components"]:
        if c["section"] not in {"CUDA_KERNELS", "APPLY_BINDINGS"} or c["kind"] not in {"kernel", "function"}:
            continue

        def add(label, line, code, component=c, **parameters):
            result.append(
                dict(
                    feature=label,
                    component=component["qualified_name"],
                    section=component["section"],
                    line=line,
                    code=code[:900],
                    reason=STRATEGIES[label]["template"],
                    parameters=parameters,
                )
            )

        if c["kind"] == "kernel":
            for symbol in sorted(constants.intersection(re.findall(r"\b\w+\b", c["source"]))):
                add("constant_memory", c["line"], c["source"], symbol=symbol)
        # cudaLaunchKernelEx receives a config object, so the cluster attribute is
        # normally assigned before the launch rather than passed as a call argument.
        # This is an explicit source observation, not a claim that the launch ran.
        if re.search(r"\bcudaLaunchAttributeClusterDimension\b", c["source"]):
            add(
                "thread_block_cluster",
                c["line"],
                c["source"],
                attribute="cudaLaunchAttributeClusterDimension",
            )
        for call in c.get("calls", []):
            target = call["target"]
            args = " ".join(call["arguments"])
            checks = {
                "cache_policy": bool(
                    re.search(
                        r"^(?:__ld(?:ca|cg|cs|cv|lu)|__st(?:wb|cg|cs|wt)|cudaStreamSetAttribute|cudaCtxResetPersistingL2Cache)$",
                        target,
                    )
                ),
                "launch_bounds": bool(
                    re.search(r"^cuda(?:FuncSetAttribute|FuncSetCacheConfig|Occupancy\w+)$", target)
                ),
                "dynamic_shared_memory": target == "cudaFuncSetAttribute"
                and "cudaFuncAttributeMaxDynamicSharedMemorySize" in args,
                "library_offload": bool(
                    re.search(
                        r"^(?:cublas(?:Lt)?(?:[SDCHZ]?gemm|Gemm|Matmul|[SDCHZ]?gemv)|cudnn(?:Convolution|Activation|Pooling|Softmax|BatchNormalization|Normalization)|cutlass::gemm)",
                        target,
                        re.I,
                    )
                ),
                "library_math_mode": target == "cublasSetMathMode"
                or (
                    target.startswith(("cublas", "cudnn"))
                    and bool(re.search(r"CUBLAS_(?:COMPUTE|TF32|TENSOR_OP)|CUDNN_(?:TENSOR_OP|FMA)_MATH", args))
                ),
                "library_algorithm": bool(
                    re.search(
                        r"(?:AlgoGetHeuristic|AlgoInit|SetConvolutionMathType|FindConvolution|GetConvolution.*Algorithm)",
                        target,
                    )
                )
                or (
                    target.startswith(("cublas", "cudnn"))
                    and bool(re.search(r"EPILOGUE|CUBLAS_GEMM_|CUDNN_.*ALGO", args))
                ),
                "cuda_graph": bool(
                    re.search(r"^cuda(?:Graph(?:Launch|Instantiate|Add\w+)|Stream(?:BeginCapture|EndCapture))", target)
                ),
                "async_transfer": bool(re.search(r"^cuda(?:Memcpy\w*Async|MemPrefetchAsync)$", target)),
                "stream_event_scheduling": bool(
                    re.search(
                        r"^cuda(?:Stream|Event)(?:Create|CreateWithFlags|Record|WaitEvent|Synchronize|Destroy)", target
                    )
                ),
                "cooperative_grid_sync": bool(re.search(r"^cudaLaunchCooperativeKernel(?:MultiDevice)?$", target)),
                "thread_block_cluster": bool(
                    re.search(r"^cuda(?:LaunchAttributeClusterDimension|FuncSetAttribute)$", target)
                    and bool(re.search(r"Cluster|NonPortableCluster", args))
                ),
                "pinned_host_memory": bool(re.search(r"^cuda(?:HostAlloc|MallocHost|HostRegister)$", target)),
                "managed_memory": bool(re.search(r"^cuda(?:MallocManaged|MemAdvise|MemPrefetchAsync)$", target)),
            }
            for label, matched in checks.items():
                if matched:
                    add(
                        label,
                        call["line"],
                        call["expression"] + "(" + ", ".join(call["arguments"]) + ")",
                        target=target,
                        arguments=call["arguments"],
                    )
    return result


def inspect_transition(before, after):
    """Compare launch sites in unique host functions; no execution-count claim."""
    result = []

    def hosts(snapshot):
        grouped = collections.defaultdict(list)
        for c in snapshot["components"]:
            if c["kind"] == "function":
                grouped[(c["section"], c["qualified_name"])].append(c)
        return {key: cs[0] for key, cs in grouped.items() if len(cs) == 1}

    old, new = hosts(before), hosts(after)
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        launches_a = [c for c in a.get("calls", []) if c["kind"] == "kernel_launch"]
        launches_b = [c for c in b.get("calls", []) if c["kind"] == "kernel_launch"]
        if not launches_a or not launches_b:
            continue

        def add(label, component=b, **parameters):
            result.append(
                dict(
                    feature=label,
                    component=component["qualified_name"],
                    section=component["section"],
                    line=component["line"],
                    code=component["source"][:900],
                    reason=STRATEGIES[label]["template"],
                    parameters=parameters,
                )
            )

        names_a, names_b = collections.Counter(c["target"] for c in launches_a), collections.Counter(
            c["target"] for c in launches_b
        )
        removed, added = list((names_a - names_b).elements()), list((names_b - names_a).elements())
        if len(launches_a) != len(launches_b) and removed and added:
            label = "launch_fusion" if len(launches_b) < len(launches_a) else "launch_fission"
            add(label, before_launches=len(launches_a), after_launches=len(launches_b), removed=removed, added=added)
            if label == "launch_fusion":
                allocated = re.findall(
                    r"(?:auto|\w+(?:::\w+)*\s*\*?)\s+(\w+)\s*=\s*[^;]*(?:empty|Empty|alloc|Alloc)\w*\s*\(", a["source"]
                )
                for buffer in allocated:
                    users = sum(
                        any(re.search(r"\b" + re.escape(buffer) + r"\b", arg) for arg in c["arguments"])
                        for c in launches_a
                    )
                    if users >= 2 and not re.search(r"\b" + re.escape(buffer) + r"\b", b["source"]):
                        add("intermediate_elimination", buffer=buffer, previous_launch_users=users)
        for target in names_a.keys() & names_b.keys():
            if names_a[target] == names_b[target] == 1:
                ca = next(c for c in launches_a if c["target"] == target)
                cb = next(c for c in launches_b if c["target"] == target)
                va, vb = [ca["expression"], ca["launch_config"]], [cb["expression"], cb["launch_config"]]
                if va != vb:
                    add("launch_config_change", target=target, before=va, after=vb)
    return result
