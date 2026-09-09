// Bounded diagnostic tracer. Serializes ordinary kernel launches; rejects graph
// capture and multiple host threads/contexts. Never use for scored timing.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "common.h"

static TraceBuffer* buffer = nullptr;
static FILE* metadata = nullptr;
static std::string output, filter;
static std::set<CUfunction> seen;
static std::recursive_mutex mutex;
static thread_local bool internal = false;
static uint32_t next_instruction = 0;
static uint64_t launch = 0, selected_count = 0;
static int cta_limit = 1, max_launches = 4;
static uint64_t capacity = 500000;
static CUcontext first_context = nullptr;
static std::thread::id host_thread;
static bool selected = false;
static bool capture_active = true;
extern "C" void runtime_trace_set_enabled(int enabled){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    capture_active=enabled!=0;
    if(metadata){fprintf(metadata,"{\"type\":\"scope\",\"enabled\":%s,\"launch\":%lu}\n",capture_active?"true":"false",launch);fflush(metadata);}
}
static void fail(const char* text) { fprintf(stderr, "RUNTIME_TRACE_ERROR %s\n", text); fflush(stderr); abort(); }
static void check_result(CUresult r,const char* call,int line) {
    if(r!=CUDA_SUCCESS){const char *name="unknown",*description="unknown";cuGetErrorName(r,&name);cuGetErrorString(r,&description);
        fprintf(stderr,"RUNTIME_TRACE_CUDA line=%d call=%s code=%d name=%s description=%s launch=%lu\n",line,call,(int)r,name,description,launch);fail("CUDA driver operation failed");}
}
#define check(call) check_result((call),#call,__LINE__)
static std::string quote(const char* s) {
    std::string out = "\"";
    for (; *s; ++s) { if (*s == '"' || *s == '\\') out += '\\'; if (*s == '\n') out += "\\n"; else out += *s; }
    return out + "\"";
}
static int env_int(const char* key, int fallback) { const char* v = getenv(key); return v ? atoi(v) : fallback; }
void nvbit_at_init() {
    const char* path = getenv("RUNTIME_TRACE_OUTPUT");
    if (!path) fail("RUNTIME_TRACE_OUTPUT required");
    output = path; filter = getenv("RUNTIME_TRACE_KERNEL") ? getenv("RUNTIME_TRACE_KERNEL") : "";
    cta_limit = env_int("RUNTIME_TRACE_CTAS", 1);
    capture_active = env_int("RUNTIME_TRACE_START_ENABLED", 1);
    max_launches = env_int("RUNTIME_TRACE_LAUNCHES", 4);
    capacity = env_int("RUNTIME_TRACE_CAPACITY", 500000);
    if (!capacity || capacity > 10000000) fail("capacity outside 1..10000000");
    metadata = fopen((output + ".jsonl").c_str(), "wx");
    if (!metadata) fail("metadata output must be new");
    fprintf(metadata, "{\"type\":\"config\",\"nvbit\":\"%s\",\"cta_limit\":%d,\"capacity\":%lu,\"max_launches\":%d,\"serialized\":true,\"filter\":%s}\n", NVBIT_VERSION, cta_limit, capacity, max_launches, quote(filter.c_str()).c_str());
    fflush(metadata);
}
void nvbit_tool_init(CUcontext ctx) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    bool previous_internal = internal;
    internal = true;
    if (first_context && first_context != ctx) fail("multiple contexts unsupported");
    if (buffer) { internal=previous_internal; return; }
    first_context = ctx;
    check(cuMemAllocManaged((CUdeviceptr*)&buffer, sizeof(TraceBuffer) + capacity*sizeof(TraceEvent), CU_MEM_ATTACH_GLOBAL));
    buffer->count = 0; buffer->capacity = capacity;
    internal = previous_internal;
}
static void instrument(CUcontext ctx, CUfunction f) {
    auto functions = nvbit_get_related_functions(ctx, f); functions.push_back(f);
    for (auto function : functions) {
        if (!seen.insert(function).second) continue;
        for (auto instr : nvbit_get_instrs(ctx, function)) {
            uint32_t id = next_instruction++;
            int mrefs = 0; const InstrType::operand_t* cbank = nullptr;
            for(int i=0;i<instr->getNumOperands();++i){const auto* op=instr->getOperand(i);if(op->type==InstrType::OperandType::CBANK && !op->u.cbank.has_reg_offset && op->u.cbank.has_imm_offset && op->nbytes>=4 && op->u.cbank.imm_offset%4==0) cbank=op;}
            for (int i=0;i<instr->getNumOperands();++i) if (instr->getOperand(i)->type == InstrType::OperandType::MREF) ++mrefs;
            fprintf(metadata, "{\"type\":\"instruction\",\"id\":%u,\"function\":%s,\"offset\":%u,\"sass\":%s,\"opcode\":%s,\"space\":%s,\"load\":%s,\"store\":%s,\"size\":%d,\"mrefs\":%d,\"predicate\":%d,\"predicate_uniform\":%s,\"operands\":[", id, quote(nvbit_get_func_name(ctx,function)).c_str(), instr->getOffset(), quote(instr->getSass()).c_str(), quote(instr->getOpcode()).c_str(), quote(InstrType::MemorySpaceStr[(int)instr->getMemorySpace()]).c_str(), instr->isLoad()?"true":"false", instr->isStore()?"true":"false", instr->getSize(), mrefs, instr->hasPred()?instr->getPredNum():-1, instr->hasPred()&&instr->isPredUniform()?"true":"false");
            for (int i=0;i<instr->getNumOperands();++i) {
                const auto* op = instr->getOperand(i);
                fprintf(metadata,"%s{\"type\":%s,\"text\":%s,\"bytes\":%d}", i?",":"",quote(InstrType::OperandTypeStr[(int)op->type]).c_str(),quote(op->str).c_str(),op->nbytes);
            }
            fprintf(metadata,"],\"captured_constant_bytes\":%d}\n",cbank?std::min(cbank->nbytes,8):0);
            nvbit_insert_call(instr,"trace_instruction",IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr,id);
            if (mrefs == 1) nvbit_add_call_arg_mref_addr64(instr,0); else nvbit_add_call_arg_const_val64(instr,0);
            if(cbank){
                nvbit_add_call_arg_cbank_val(instr,cbank->u.cbank.id,cbank->u.cbank.imm_offset);
                if(cbank->nbytes>=8)nvbit_add_call_arg_cbank_val(instr,cbank->u.cbank.id,cbank->u.cbank.imm_offset+4);else nvbit_add_call_arg_const_val32(instr,0);
            }else{nvbit_add_call_arg_const_val32(instr,0);nvbit_add_call_arg_const_val32(instr,0);}
            nvbit_add_call_arg_const_val64(instr,(uint64_t)buffer);
            nvbit_add_call_arg_const_val32(instr,(uint32_t)cta_limit);
        }
    }
    fflush(metadata);
}
void nvbit_at_cuda_event(CUcontext ctx,int is_exit,nvbit_api_cuda_t cbid,const char* name,void* params,CUresult* status) {
    if (internal) return;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    internal=true;
    if (!capture_active) {
        if(!is_exit && (cbid==API_CUDA_cuLaunchKernel || cbid==API_CUDA_cuLaunchKernel_ptsz))nvbit_enable_instrumented(ctx,((cuLaunchKernel_params*)params)->f,false);
        if(!is_exit && (cbid==API_CUDA_cuLaunchKernelEx || cbid==API_CUDA_cuLaunchKernelEx_ptsz))nvbit_enable_instrumented(ctx,((cuLaunchKernelEx_params*)params)->f,false);
        internal=false; return;
    }
    if (strstr(name,"cuGraphLaunch") || strstr(name,"cuStreamBeginCapture")) fail("CUDA Graph capture/launch unsupported");
    bool ordinary = cbid==API_CUDA_cuLaunchKernel || cbid==API_CUDA_cuLaunchKernel_ptsz;
    bool extended = cbid==API_CUDA_cuLaunchKernelEx || cbid==API_CUDA_cuLaunchKernelEx_ptsz;
    if (!ordinary && !extended) {
        if (!is_exit && (strstr(name,"cuMemcpy") || strstr(name,"cuMemset") || strstr(name,"cuMemFree") || strstr(name,"cuMemAlloc"))) {
            fprintf(metadata,"{\"type\":\"memory_api\",\"before_launch\":%lu,\"name\":%s}\n",launch,quote(name).c_str());fflush(metadata);
        }
        if (!is_exit && (strstr(name,"cuLaunch") || strstr(name,"cuGraph"))) {
            fprintf(metadata,"{\"type\":\"unsupported_api\",\"name\":%s}\n",quote(name).c_str());fflush(metadata);
        }
        internal=false;return;
    }
    if (host_thread==std::thread::id()) host_thread=std::this_thread::get_id();
    if (host_thread!=std::this_thread::get_id()) fail("multiple launch host threads unsupported");
    CUfunction function; CUstream stream; unsigned gx,gy,gz,bx,by,bz,shared;
    if (ordinary) {
        auto* p=(cuLaunchKernel_params*)params;function=p->f;stream=p->hStream;
        gx=p->gridDimX;gy=p->gridDimY;gz=p->gridDimZ;bx=p->blockDimX;by=p->blockDimY;bz=p->blockDimZ;shared=p->sharedMemBytes;
    } else {
        auto* p=(cuLaunchKernelEx_params*)params;function=p->f;auto* c=p->config;stream=c->hStream;
        gx=c->gridDimX;gy=c->gridDimY;gz=c->gridDimZ;bx=c->blockDimX;by=c->blockDimY;bz=c->blockDimZ;shared=c->sharedMemBytes;
    }
    if (first_context && first_context != ctx) fail("multiple active contexts unsupported");
    if (!is_exit) {
        if (!buffer) { nvbit_tool_init(ctx); internal=true; }
        check(cuCtxSynchronize());
        const char* fname=nvbit_get_func_name(ctx,function);
        selected=(max_launches<0 || selected_count<(uint64_t)max_launches) && strstr(fname,filter.c_str());
        fprintf(metadata,"{\"type\":\"launch\",\"id\":%lu,\"name\":%s,\"stream\":%lu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"dynamic_shared\":%u,\"selected\":%s}\n",launch,quote(fname).c_str(),(uint64_t)stream,gx,gy,gz,bx,by,bz,shared,selected?"true":"false");
        if(selected){instrument(ctx,function);buffer->count=0;++selected_count;}
        nvbit_enable_instrumented(ctx,function,selected);
    } else {
        check(*status);check(cuCtxSynchronize());
        if(selected){
            uint64_t count=std::min((uint64_t)buffer->count,capacity);
            std::string path=output+".launch"+std::to_string(launch)+".bin";
            FILE* f=fopen(path.c_str(),"wbx");if(!f)fail("binary output must be new");
            if(fwrite(buffer->events,sizeof(TraceEvent),count,f)!=count)fail("trace write failed");
            if(fclose(f))fail("trace close failed");
            fprintf(metadata,"{\"type\":\"complete\",\"launch\":%lu,\"events\":%lu,\"attempted\":%llu,\"dropped\":%llu}\n",launch,count,buffer->count,buffer->count-count);
        }
        ++launch;
    }
    fflush(metadata);internal=false;
}
void nvbit_at_term(){if(metadata){fprintf(metadata,"{\"type\":\"end\",\"launches\":%lu}\n",launch);fclose(metadata);}}
