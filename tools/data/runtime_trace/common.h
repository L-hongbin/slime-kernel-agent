#pragma once
#include <stdint.h>
struct TraceEvent {
    uint64_t address, constant;
    uint32_t instruction, cta, thread, predicate, active_mask;
    uint32_t pad;
};
struct TraceBuffer {
    unsigned long long count;
    unsigned long long capacity;
    TraceEvent events[1];
};

static_assert(sizeof(TraceEvent)==40,"Python event decoder layout must match");
