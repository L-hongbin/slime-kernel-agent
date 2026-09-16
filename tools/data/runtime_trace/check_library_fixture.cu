// Calibration fixture for generic cuBLAS strided-batched GEMM contracts.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>

#define CUDA(call)                                                                                                   \
    do {                                                                                                             \
        cudaError_t error = (call);                                                                                  \
        if (error != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));                                         \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

#define CUBLAS(call)                                                                                                 \
    do {                                                                                                             \
        cublasStatus_t error = (call);                                                                               \
        if (error != CUBLAS_STATUS_SUCCESS) {                                                                        \
            std::fprintf(stderr, "cuBLAS: %d\n", static_cast<int>(error));                                        \
            return 3;                                                                                                \
        }                                                                                                            \
    } while (0)

int main() {
    constexpr int matrix = 4;
    constexpr int batches = 2;
    float a_host[batches * matrix] = {1, 3, 2, 4, 2, 0, 1, 2};
    float b_host[batches * matrix] = {5, 7, 6, 8, 1, 3, 4, 2};
    float c_host[batches * matrix] = {};
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA(cudaMalloc(&a, sizeof(a_host)));
    CUDA(cudaMalloc(&b, sizeof(b_host)));
    CUDA(cudaMalloc(&c, sizeof(c_host)));
    CUDA(cudaMemcpy(a, a_host, sizeof(a_host), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, b_host, sizeof(b_host), cudaMemcpyHostToDevice));
    CUDA(cudaMemset(c, 0, sizeof(c_host)));
    std::printf(
        "{\"allocations\":[{\"role\":\"input:a\",\"base\":%llu,\"bytes\":%zu},"
        "{\"role\":\"input:b\",\"base\":%llu,\"bytes\":%zu},"
        "{\"role\":\"output:0\",\"base\":%llu,\"bytes\":%zu}]}\n",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(a)), sizeof(a_host),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(b)), sizeof(b_host),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(c)), sizeof(c_host));
    cublasHandle_t handle = nullptr;
    CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    auto enable = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "runtime_trace_set_enabled"));
    if (enable) enable(1);
    // Two independent 2x2 column-major products with element strides 4.
    CUBLAS(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, a, 2, matrix, b, 2, matrix, &beta, c, 2, matrix, batches));
    // alpha=0 proves the contract must not read A/B; stride=0 is still valid
    // metadata and beta=1 reads each pre-existing C batch before writing it.
    alpha = 0.0f;
    beta = 1.0f;
    CUBLAS(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, a, 2, 0, b, 2, 0, &beta, c, 2, matrix, batches));
    CUDA(cudaDeviceSynchronize());
    if (enable) enable(0);
    CUDA(cudaMemcpy(c_host, c, sizeof(c_host), cudaMemcpyDeviceToHost));
    std::printf("{\"output\":[");
    for (int index = 0; index < batches * matrix; ++index) {
        std::printf("%s%.9g", index ? "," : "", c_host[index]);
    }
    std::printf("]}\n");
    CUBLAS(cublasDestroy(handle));
    CUDA(cudaFree(a));
    CUDA(cudaFree(b));
    CUDA(cudaFree(c));
    return 0;
}
