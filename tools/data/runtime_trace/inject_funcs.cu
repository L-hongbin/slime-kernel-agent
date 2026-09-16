#include "common.h"
#include "utils/utils.h"
#include <cuda/atomic>
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
    TraceBuffer* b=(TraceBuffer*)buffer_pointer;
    // Once full, stop contending on a global counter. Lost records become an
    // explicit lower bound, never an exact sampled population estimate.
    if(((volatile TraceBuffer*)b)->count>=b->capacity){
        if(!((volatile TraceBuffer*)b)->truncated)atomicExch(&b->truncated,1u);
        return;
    }
    auto index=atomicAdd(&b->count,1ULL);
    if(index<b->capacity)b->runs[index]={address,owner,width*uint32_t(coalesced?count:1),width,mode,0};
}

extern "C" __device__ __noinline__ void trace_regions(int pred,uint64_t address,
    uint32_t width,uint32_t mode,uint64_t buffer_pointer,int cta_limit){
    if(!pred)return;
    uint64_t cta=blockIdx.x+uint64_t(gridDim.x)*(blockIdx.y+uint64_t(gridDim.y)*blockIdx.z);
    if(cta_limit>=0&&cta>=uint64_t(cta_limit))return;
    unsigned mask=__activemask();int lane=get_laneid(),first=__ffs(mask)-1,count=__popc(mask);
    uint64_t base=__shfl_sync(mask,address,first);
    bool affine=address==base+uint64_t(lane-first)*width;
    bool contiguous=(mask>>first)==(count==32?0xffffffffu:((1u<<count)-1));
    bool coalesced=contiguous&&__all_sync(mask,affine);
    unsigned same=__match_any_sync(mask,address);
    bool duplicate_writer=(mode&2)&&__popc(same)>1;
    if(coalesced){if(lane!=first)return;width*=count;}
    else if(lane!=__ffs(same)-1)return;  // Broadcasts retain one exact address.
    RegionBuffer* b=(RegionBuffer*)buffer_pointer;
    while(width){
        uint64_t key=address/REGION_BYTES+1;
        uint64_t at=((key^(key>>33))*0xff51afd7ed558ccdULL)&(b->capacity-1);
        MemoryTile* tile=nullptr;
        for(unsigned probe=0;probe<128;++probe){
            MemoryTile* candidate=&b->tiles[(at+probe)&(b->capacity-1)];
            auto found=cuda::atomic_ref<unsigned long long,cuda::thread_scope_device>(candidate->key).load(cuda::memory_order_relaxed);
            if(!found)found=atomicCAS(&candidate->key,0ULL,key);
            if(found==0||found==key){tile=candidate;break;}
        }
        if(!tile){atomicExch(&b->overflow,1u);return;}
        unsigned offset=address%REGION_BYTES,word=offset/64,bit=offset%64;
        unsigned n=min(width,64u-bit);
        unsigned long long bits=(n==64?~0ULL:((1ULL<<n)-1))<<bit;
        if(mode&1){
            auto present=cuda::atomic_ref<unsigned long long,cuda::thread_scope_device>(tile->reads[word]).load(cuda::memory_order_relaxed);
            if((present&bits)!=bits)atomicOr(&tile->reads[word],bits);
        }
        if(mode&2){
            auto old=atomicOr(&tile->writes[word],bits);
            if((old&bits)||duplicate_writer)atomicOr(&tile->flags,2ULL);
        }
        if(mode==3)atomicOr(&tile->flags,1ULL);
        address+=n;width-=n;
    }
}

// The same CUfunction may be launched repeatedly with different capture
// precision. Its instrumentation stays fixed; the host selects per launch.
extern "C" __device__ __noinline__ void trace_dispatch(int pred,uint64_t address,
    uint32_t width,uint32_t mode,uint64_t pointer,int cta_limit){
    CaptureDispatch* dispatch=(CaptureDispatch*)pointer;
    if(dispatch->ordered)trace_memory(pred,address,width,mode,dispatch->runs,cta_limit);
    else trace_regions(pred,address,width,mode,dispatch->regions,cta_limit);
}
