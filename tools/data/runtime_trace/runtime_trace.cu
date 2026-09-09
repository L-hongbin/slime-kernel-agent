// Whole-forward coarse capture: global-memory runs, opaque kernel versions,
// runtime launch parameters and library parent scopes. No register graph.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "common.h"
static TraceBuffer* buffer=nullptr;
static FILE* log_file=nullptr;
static std::recursive_mutex lock;
static thread_local bool internal=false;
static thread_local long long library_parent=-1;
static CUcontext first_context=nullptr;
static std::thread::id launch_thread;
static std::string prefix;
static bool active=true,selected=false;
static uint64_t position=0,launch_id=0,capacity=1048576;
static uint64_t remaining_records=~0ULL;
static int max_launches=-1;
static bool memory_enabled=true,skip_vendor=false;
static const char* skip_reason="";
static int cta_limit=-1;
struct FunctionInfo {uint64_t hash=1469598103934665603ULL;int unsupported=0;};
static std::unordered_map<CUfunction,FunctionInfo> info;
static void fail(const char* message){fprintf(stderr,"COARSE_TRACE_ERROR %s\n",message);abort();}
static void check_result(CUresult status,const char* expr){if(status!=CUDA_SUCCESS){const char *name=nullptr,*msg=nullptr;cuGetErrorName(status,&name);cuGetErrorString(status,&msg);fprintf(stderr,"COARSE_CUDA %s %s %s\n",expr,name,msg);fail("CUDA operation failed");}}
#define CHECK(call) check_result((call),#call)
static std::string quote(const char* s){std::string out="\"";for(;*s;++s){if(*s=='"'||*s=='\\')out+='\\';if(*s=='\n')out+="\\n";else out+=*s;}return out+'"';}
static void flush(){if(fflush(log_file)||ferror(log_file))fail("metadata write failed");}
static int env_int(const char* key,int fallback){const char* p=getenv(key);return p?atoi(p):fallback;}
extern "C" uint64_t coarse_position(){return position;}
extern "C" int coarse_active(){return active;}
extern "C" void runtime_trace_set_enabled(int value){std::lock_guard<std::recursive_mutex> guard(lock);active=value!=0;if(log_file){fprintf(log_file,"{\"type\":\"scope\",\"seq\":%lu,\"enabled\":%s}\n",position++,active?"true":"false");flush();}}
extern "C" long long coarse_library_begin(const char* api,const char* attributes,uint64_t stream){
    std::lock_guard<std::recursive_mutex> guard(lock);if(!active||library_parent>=0)return -1;long long id=position++;
    fprintf(log_file,"{\"type\":\"library_begin\",\"seq\":%lld,\"api\":%s,\"stream\":%lu,\"attrs\":%s}\n",id,quote(api).c_str(),stream,attributes);library_parent=id;flush();return id;
}
extern "C" void coarse_library_end(long long id,int status){std::lock_guard<std::recursive_mutex> guard(lock);if(id<0)return;fprintf(log_file,"{\"type\":\"library_end\",\"seq\":%lu,\"parent\":%lld,\"status\":%d}\n",position++,id,status);library_parent=-1;flush();}
void nvbit_at_init(){
    const char* p=getenv("RUNTIME_TRACE_OUTPUT");if(!p)fail("RUNTIME_TRACE_OUTPUT required");prefix=p;
    active=env_int("RUNTIME_TRACE_START_ENABLED",1);cta_limit=env_int("RUNTIME_TRACE_CTAS",-1);capacity=env_int("RUNTIME_TRACE_CAPACITY",1048576);
    max_launches=env_int("RUNTIME_TRACE_MAX_LAUNCHES",-1);
    int total=env_int("RUNTIME_TRACE_TOTAL_RECORDS",-1);if(total>=0)remaining_records=total;
    const char* policy=getenv("RUNTIME_TRACE_MEMORY_POLICY");memory_enabled=!policy||strcmp(policy,"interfaces");
    skip_vendor=env_int("RUNTIME_TRACE_SKIP_VENDOR_MEMORY",0);
    if(capacity<1||capacity>16777216)fail("memory run capacity out of range");
    log_file=fopen((prefix+".jsonl").c_str(),"wx");if(!log_file)fail("trace output must be new");
    fprintf(log_file,"{\"type\":\"config\",\"schema\":\"coarse-memory-runs/v1\",\"nvbit\":\"%s\",\"cta_limit\":%d,\"capacity\":%lu,\"serialized_diagnostic\":true,\"max_launches\":%d,\"total_records\":%lld,\"memory_enabled\":%s,\"skip_vendor_memory\":%s,\"counter_mode\":\"saturating_lower_bound\"}\n",NVBIT_VERSION,cta_limit,capacity,max_launches,(long long)remaining_records,memory_enabled?"true":"false",skip_vendor?"true":"false");flush();
}
void nvbit_tool_init(CUcontext ctx){std::lock_guard<std::recursive_mutex> guard(lock);bool old=internal;internal=true;
    if(first_context&&first_context!=ctx)fail("multiple contexts unsupported");first_context=ctx;
    if(!buffer){CHECK(cuMemAllocManaged((CUdeviceptr*)&buffer,sizeof(TraceBuffer)+capacity*sizeof(MemoryRun),CU_MEM_ATTACH_GLOBAL));buffer->capacity=capacity;buffer->count=0;}internal=old;
}
static void instrument(CUcontext ctx,CUfunction function){
    auto funcs=nvbit_get_related_functions(ctx,function);funcs.push_back(function);
    for(auto f:funcs){if(info.count(f))continue;FunctionInfo fi;
        for(auto ins:nvbit_get_instrs(ctx,f)){
            for(const char* p=ins->getSass();*p;++p){fi.hash^=(unsigned char)*p;fi.hash*=1099511628211ULL;}
            auto space=ins->getMemorySpace();
            if(space==InstrType::MemorySpace::NONE||space==InstrType::MemorySpace::CONSTANT||space==InstrType::MemorySpace::SHARED||space==InstrType::MemorySpace::LOCAL)continue;
            int refs=0;for(int i=0;i<ins->getNumOperands();++i)refs+=ins->getOperand(i)->type==InstrType::OperandType::MREF;
            if(space!=InstrType::MemorySpace::GLOBAL||refs!=1||ins->getSize()<=0||(!ins->isLoad()&&!ins->isStore())){++fi.unsupported;continue;}
            nvbit_insert_call(ins,"trace_memory",IPOINT_BEFORE);nvbit_add_call_arg_guard_pred_val(ins);nvbit_add_call_arg_mref_addr64(ins,0);
            int mode=(ins->isLoad()?1:0)|(ins->isStore()?2:0);
            if(strstr(ins->getOpcode(),"ATOM")||strncmp(ins->getOpcode(),"RED",3)==0)mode=3;
            nvbit_add_call_arg_const_val32(ins,ins->getSize());nvbit_add_call_arg_const_val32(ins,mode);
            nvbit_add_call_arg_const_val64(ins,(uint64_t)buffer);nvbit_add_call_arg_const_val32(ins,(uint32_t)cta_limit);
        }info[f]=fi;
    }
}
static FunctionInfo combined_info(CUcontext ctx,CUfunction function){
    auto functions=nvbit_get_related_functions(ctx,function);functions.push_back(function);
    std::vector<uint64_t> hashes;FunctionInfo result;
    for(auto f:functions){hashes.push_back(info[f].hash);result.unsupported+=info[f].unsupported;}
    std::sort(hashes.begin(),hashes.end());
    for(auto hash:hashes)for(int i=0;i<8;++i){result.hash^=(hash>>(i*8))&255;result.hash*=1099511628211ULL;}
    return result;
}
void nvbit_at_cuda_event(CUcontext ctx,int exiting,nvbit_api_cuda_t cbid,const char* name,void* params,CUresult* status){
    if(internal)return;std::lock_guard<std::recursive_mutex> guard(lock);internal=true;
    bool ordinary=cbid==API_CUDA_cuLaunchKernel||cbid==API_CUDA_cuLaunchKernel_ptsz;
    bool extended=cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz;
    if(!ordinary&&!extended){
        if(active&&!exiting){
            if(strstr(name,"cuGraphLaunch")||strstr(name,"cuStreamBeginCapture"))fail("CUDA Graph capture unsupported");
            if(strstr(name,"cuLaunch")){fprintf(log_file,"{\"type\":\"unsupported_launch\",\"seq\":%lu,\"name\":%s}\n",position++,quote(name).c_str());flush();}
        }
        if(active&&exiting&&(strstr(name,"cuMemcpy")||strstr(name,"cuMemset"))){
            if(*status==CUDA_SUCCESS&&(cbid==API_CUDA_cuMemcpyDtoDAsync_v2||cbid==API_CUDA_cuMemcpyDtoDAsync_v2_ptsz)){auto* p=(cuMemcpyDtoDAsync_v2_params*)params;fprintf(log_file,"{\"type\":\"copy\",\"seq\":%lu,\"source\":%lu,\"target\":%lu,\"bytes\":%lu,\"stream\":%lu}\n",position++,(uint64_t)p->srcDevice,(uint64_t)p->dstDevice,p->ByteCount,(uint64_t)p->hStream);}
            else if(*status==CUDA_SUCCESS&&(cbid==API_CUDA_cuMemcpyAsync||cbid==API_CUDA_cuMemcpyAsync_ptsz)){auto* p=(cuMemcpyAsync_params*)params;fprintf(log_file,"{\"type\":\"copy\",\"seq\":%lu,\"source\":%lu,\"target\":%lu,\"bytes\":%lu,\"stream\":%lu}\n",position++,(uint64_t)p->src,(uint64_t)p->dst,p->ByteCount,(uint64_t)p->hStream);}
            else if(*status==CUDA_SUCCESS&&(cbid==API_CUDA_cuMemsetD8Async||cbid==API_CUDA_cuMemsetD8Async_ptsz)){auto* p=(cuMemsetD8Async_params*)params;fprintf(log_file,"{\"type\":\"fill\",\"seq\":%lu,\"target\":%lu,\"bytes\":%lu,\"value\":%u,\"element_bytes\":1,\"stream\":%lu}\n",position++,(uint64_t)p->dstDevice,p->N,(unsigned)p->uc,(uint64_t)p->hStream);}
            else if(*status==CUDA_SUCCESS&&(cbid==API_CUDA_cuMemsetD32Async||cbid==API_CUDA_cuMemsetD32Async_ptsz)){auto* p=(cuMemsetD32Async_params*)params;fprintf(log_file,"{\"type\":\"fill\",\"seq\":%lu,\"target\":%lu,\"bytes\":%lu,\"value\":%u,\"element_bytes\":4,\"stream\":%lu}\n",position++,(uint64_t)p->dstDevice,p->N*4,(unsigned)p->ui,(uint64_t)p->hStream);}
            else fprintf(log_file,"{\"type\":\"memory_api_unknown\",\"seq\":%lu,\"name\":%s,\"status\":%d}\n",position++,quote(name).c_str(),(int)*status);flush();
        }
        if(active&&exiting&&library_parent<0&&*status==CUDA_SUCCESS){
            if(cbid==API_CUDA_cuEventRecord||cbid==API_CUDA_cuEventRecord_ptsz){auto* p=(cuEventRecord_params*)params;fprintf(log_file,"{\"type\":\"order\",\"action\":\"record\",\"seq\":%lu,\"event\":%lu,\"stream\":%lu}\n",position++,(uint64_t)p->hEvent,(uint64_t)p->hStream);flush();}
            else if(cbid==API_CUDA_cuStreamWaitEvent||cbid==API_CUDA_cuStreamWaitEvent_ptsz){auto* p=(cuStreamWaitEvent_params*)params;fprintf(log_file,"{\"type\":\"order\",\"action\":\"wait\",\"seq\":%lu,\"event\":%lu,\"stream\":%lu}\n",position++,(uint64_t)p->hEvent,(uint64_t)p->hStream);flush();}
            else if(cbid==API_CUDA_cuEventDestroy_v2){auto* p=(cuEventDestroy_v2_params*)params;fprintf(log_file,"{\"type\":\"order\",\"action\":\"destroy\",\"seq\":%lu,\"event\":%lu}\n",position++,(uint64_t)p->hEvent);flush();}
            else if(cbid==API_CUDA_cuCtxSynchronize){fprintf(log_file,"{\"type\":\"order\",\"action\":\"device_sync\",\"seq\":%lu}\n",position++);flush();}
        }
        internal=false;return;
    }
    CUfunction f;CUstream stream;void** arguments;unsigned gx,gy,gz,bx,by,bz,shared;
    if(ordinary){auto* p=(cuLaunchKernel_params*)params;f=p->f;stream=p->hStream;arguments=p->kernelParams;gx=p->gridDimX;gy=p->gridDimY;gz=p->gridDimZ;bx=p->blockDimX;by=p->blockDimY;bz=p->blockDimZ;shared=p->sharedMemBytes;}
    else{auto* p=(cuLaunchKernelEx_params*)params;f=p->f;arguments=p->kernelParams;auto* c=p->config;stream=c->hStream;gx=c->gridDimX;gy=c->gridDimY;gz=c->gridDimZ;bx=c->blockDimX;by=c->blockDimY;bz=c->blockDimZ;shared=c->sharedMemBytes;}
    if(!active){if(!exiting)nvbit_enable_instrumented(ctx,f,false);internal=false;return;}
    if(max_launches>=0&&launch_id>=uint64_t(max_launches)){
        if(!exiting){nvbit_enable_instrumented(ctx,f,false);if(launch_id==uint64_t(max_launches)){fprintf(log_file,"{\"type\":\"unsupported_launch\",\"seq\":%lu,\"name\":\"launch_budget_exhausted\"}\n",position++);flush();}}
        else ++launch_id;internal=false;return;
    }
    if(first_context&&first_context!=ctx)fail("multiple active CUDA contexts unsupported");
    if(launch_thread==std::thread::id())launch_thread=std::this_thread::get_id();if(launch_thread!=std::this_thread::get_id())fail("multiple host launch threads unsupported");
    if(!exiting){
        if(!buffer)nvbit_tool_init(ctx);internal=true;CHECK(cuCtxSynchronize());
        const char* symbol=nvbit_get_func_name(ctx,f);
        bool vendor=skip_vendor&&(strstr(symbol,"cudnn")||strstr(symbol,"cublas")||strstr(symbol,"cutlass::")||strstr(symbol,"xmma_")||strstr(symbol,"gemv2"));
        skip_reason=library_parent>=0?"library_api_contract":!memory_enabled?"interface_only_policy":vendor?"vendor_memory_opaque":remaining_records==0?"record_budget_exhausted":"";
        selected=!skip_reason[0];if(selected)instrument(ctx,f);
        if(selected){buffer->count=0;buffer->capacity=std::min(capacity,remaining_records);buffer->truncated=0;}
        FunctionInfo aggregate=selected?combined_info(ctx,f):FunctionInfo{};
        fprintf(log_file,"{\"type\":\"launch\",\"seq\":%lu,\"id\":%lu,\"parent\":%lld,\"name\":%s,\"stream\":%lu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared\":%u,\"memory_traced\":%s,\"implementation\":\"%016lx\",\"unsupported_memory_instructions\":%d,\"arguments\":[",position++,launch_id,library_parent,quote(nvbit_get_func_name(ctx,f)).c_str(),(uint64_t)stream,gx,gy,gz,bx,by,bz,shared,selected?"true":"false",selected?aggregate.hash:0,selected?aggregate.unsupported:0);
        bool inspect_arguments=!(skip_vendor&&(vendor||library_parent>=0));
        auto sizes=inspect_arguments?nvbit_get_kernel_argument_sizes(ctx,f):std::vector<int>{};
        if(arguments)for(size_t i=0;i<sizes.size();++i){uint64_t value=0;if(sizes[i]<=8)memcpy(&value,arguments[i],sizes[i]);fprintf(log_file,"%s{\"index\":%lu,\"bytes\":%d,\"bits\":%lu}",i?",":"",i,sizes[i],value);}
        unsigned num_attrs=ordinary?0:((cuLaunchKernelEx_params*)params)->config->numAttrs;
        fprintf(log_file,"],\"arguments_present\":%s,\"launch_api\":\"%s\",\"launch_num_attrs\":%u,\"memory_skip_reason\":%s,\"implementation_version\":\"%016lx\",\"implementation_inspected\":%s}\n",inspect_arguments&&(arguments||sizes.empty())?"true":"false",ordinary?"cuLaunchKernel":"cuLaunchKernelEx",num_attrs,quote(skip_reason).c_str(),aggregate.hash,selected?"true":"false");nvbit_enable_instrumented(ctx,f,selected);flush();
    }else{
        CHECK(*status);CHECK(cuCtxSynchronize());uint64_t kept=selected?std::min((uint64_t)buffer->count,(uint64_t)buffer->capacity):0;
        if(selected)remaining_records-=kept;
        if(selected){std::string path=prefix+".launch"+std::to_string(launch_id)+".bin";FILE* f=fopen(path.c_str(),"wbx");if(!f)fail("binary must be new");if(fwrite(buffer->runs,sizeof(MemoryRun),kept,f)!=kept||fclose(f))fail("binary write failed");}
        fprintf(log_file,"{\"type\":\"complete\",\"seq\":%lu,\"id\":%lu,\"runs\":%lu,\"dropped\":%lu,\"drop_count_exact\":%s}\n",position++,launch_id,kept,selected?(uint64_t)buffer->count-kept+buffer->truncated:0,selected&&buffer->truncated?"false":"true");++launch_id;flush();
    }internal=false;
}
void nvbit_at_term(){if(log_file){fprintf(log_file,"{\"type\":\"end\",\"seq\":%lu,\"launches\":%lu}\n",position++,launch_id);flush();fclose(log_file);}}
