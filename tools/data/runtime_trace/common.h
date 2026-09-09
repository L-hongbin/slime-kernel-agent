#pragma once
#include <stdint.h>
// Exact byte range with affine thread ownership. Kernel internals remain opaque.
struct MemoryRun {
    uint64_t address, owner;
    uint32_t bytes, width, mode, reserved;
};
struct TraceBuffer {
    unsigned long long count, capacity;
    unsigned int truncated;
    MemoryRun runs[1];
};
static_assert(sizeof(MemoryRun)==32,"memory run decoder layout");
