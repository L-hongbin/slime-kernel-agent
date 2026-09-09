#include "common.h"
#include "utils/utils.h"
extern "C" __device__ __noinline__ void trace_memory(int pred,uint64_t address,
    uint32_t width,uint32_t mode,uint64_t buffer_pointer,int cta_limit){
    if(!pred)return;
    uint64_t cta=blockIdx.x+uint64_t(gridDim.x)*(blockIdx.y+uint64_t(gridDim.y)*blockIdx.z);
    if(cta_limit>=0&&cta>=uint64_t(cta_limit))return;
    unsigned mask=__activemask();int lane=get_laneid(),first=__ffs(mask)-1,count=__popc(mask);
    uint64_t base=__shfl_sync(mask,address,first);
    bool affine=address==base+uint64_t(lane-first)*width;
    bool contiguous=(mask>>first)==(count==32?0xffffffffu:((1u<<count)-1));
    bool coalesced=contiguous&&__all_sync(mask,affine);
    uint64_t tid=threadIdx.x+uint64_t(blockDim.x)*(threadIdx.y+uint64_t(blockDim.y)*threadIdx.z);
    uint64_t owner=cta*uint64_t(blockDim.x)*blockDim.y*blockDim.z+tid;
    if(coalesced&&lane!=first)return;
    TraceBuffer* b=(TraceBuffer*)buffer_pointer;auto index=atomicAdd(&b->count,1ULL);
    if(index<b->capacity)b->runs[index]={address,owner,width*uint32_t(coalesced?count:1),width,mode,0};
}
