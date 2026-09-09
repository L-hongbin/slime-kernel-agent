#include "common.h"
#include "utils/utils.h"
extern "C" __device__ __noinline__ void trace_instruction(
    int pred, uint32_t instruction, uint64_t address, uint32_t constant_low, uint32_t constant_high, uint64_t buffer_ptr,
    int cta_limit) {
    uint32_t cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (cta_limit >= 0 && cta >= (uint32_t)cta_limit) return;
    TraceBuffer* buffer = (TraceBuffer*)buffer_ptr;
    unsigned mask = __activemask();
    auto index = atomicAdd(&buffer->count, 1ULL);
    if (index >= buffer->capacity) return;
    TraceEvent& event = buffer->events[index];
    event.address = address;
    event.constant = (uint64_t)constant_low | ((uint64_t)constant_high << 32);
    event.instruction = instruction;
    event.cta = cta;
    event.thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    event.predicate = pred;
    event.active_mask = mask;
    event.pad = 0;
}
