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

// Exact byte masks, reset for each launch. Spatial tiles are only an index:
// a set bit represents an observed byte, including holes within a tile.
constexpr unsigned REGION_BYTES = 1024;
constexpr unsigned REGION_WORDS = REGION_BYTES / 64;
struct MemoryTile {
    unsigned long long key;  // (base / REGION_BYTES) + 1; zero is unused
    unsigned long long reads[REGION_WORDS], writes[REGION_WORDS];
    unsigned long long flags;  // 1: atomic; 2: overlapping writes
};
struct RegionBuffer {
    unsigned long long capacity;
    unsigned int overflow;
    MemoryTile tiles[1];
};
struct CaptureDispatch {
    uint64_t runs, regions;
    uint32_t ordered;
};
