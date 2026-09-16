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
#include <unordered_set>
#include <vector>
#include <sys/stat.h>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "common.h"
static TraceBuffer* buffer=nullptr;
static RegionBuffer* region_buffer=nullptr;
static CaptureDispatch* dispatch=nullptr;
static std::unordered_set<uint64_t> ordered_launches;
static bool ordered_launch=false;
static uint64_t remaining_ordered_records=0;
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
static bool aggregate_regions=false;
static const char* skip_reason="";
static int cta_limit=-1;
static int max_library_module_dumps=16,max_library_module_bytes=64*1024*1024;
struct FunctionInfo {uint64_t hash=1469598103934665603ULL;int unsupported=0;};
static std::unordered_map<CUfunction,FunctionInfo> info;
static std::unordered_set<CUfunction> instrumented;
struct LibraryModule {int id=-1;bool attempted=false,complete=false,dump_api_called=false,dump_api_return=false;uint64_t bytes=0;std::string reason;};
static std::unordered_map<CUmodule,LibraryModule> library_modules;
static int next_library_module_id=0;
static void fail(const char* message){fprintf(stderr,"COARSE_TRACE_ERROR %s\n",message);abort();}
static void check_result(CUresult status,const char* expr){if(status!=CUDA_SUCCESS){const char *name=nullptr,*msg=nullptr;cuGetErrorName(status,&name);cuGetErrorString(status,&msg);fprintf(stderr,"COARSE_CUDA %s %s %s\n",expr,name,msg);fail("CUDA operation failed");}}
#define CHECK(call) check_result((call),#call)
static std::string quote(const char* s){std::string out="\"";for(;*s;++s){if(*s=='"'||*s=='\\')out+='\\';if(*s=='\n')out+="\\n";else out+=*s;}return out+'"';}
static void flush(){if(fflush(log_file)||ferror(log_file))fail("metadata write failed");}
static int env_int(const char* key,int fallback){const char* p=getenv(key);return p?atoi(p):fallback;}
static bool sane_cuda_elf(const std::string& path){
    // NVBit 1.8's dump API returns false even after it has written a valid
    // cubin.  Trust neither that return nor a filename: accept only a bounded
    // on-disk 64-bit little-endian CUDA ELF header.  Extract performs the
    // stronger section/entry validation before using its hash for credit.
    unsigned char header[64]{};FILE* file=fopen(path.c_str(),"rb");
    if(!file)return false;size_t bytes=fread(header,1,sizeof(header),file);fclose(file);
    return bytes==sizeof(header)&&header[0]==0x7f&&header[1]=='E'&&header[2]=='L'&&header[3]=='F'&&header[4]==2&&header[5]==1&&header[18]==0xbe&&header[19]==0x00;
}
static std::string library_binary(CUcontext ctx,CUfunction function,bool& complete){
    complete=false;CUmodule module=nullptr;
    if(cuFuncGetModule(&module,function)!=CUDA_SUCCESS||!module)return "{\"complete\":false,\"reason\":\"module_lookup_failed\"}";
    auto& evidence=library_modules[module];
    if(!evidence.attempted){
        evidence.attempted=true;
        // This is a trace-wide attempt budget, not the number of currently
        // cached handles: unload/reload cycles must not buy more dump budget.
        if(next_library_module_id>=max_library_module_dumps)evidence.reason="module_dump_budget_exceeded";
        else{
            // CUmodule handles may be reused after cuModuleUnload.  IDs are
            // never reused within this trace, so a stale path cannot become
            // evidence for a newly loaded module.
            evidence.id=next_library_module_id++;
            std::string path=prefix+".module"+std::to_string(evidence.id)+".cubin";
            struct stat info{};
            if(lstat(path.c_str(),&info)==0)evidence.reason="module_dump_path_already_exists";
            else{
                evidence.dump_api_called=true;evidence.dump_api_return=nvbit_dump_cubin(ctx,function,path.c_str());
                evidence.reason=evidence.dump_api_return?"api_return_true":"api_return_false";
            }
            if(evidence.reason=="module_dump_path_already_exists"){}
            else if(stat(path.c_str(),&info)||info.st_size<64||(uint64_t)info.st_size>(uint64_t)max_library_module_bytes){
                evidence.reason="module_dump_size_invalid_or_exceeded";std::remove(path.c_str());
            }else if(!sane_cuda_elf(path)){evidence.reason="module_dump_elf_sanity_failed";std::remove(path.c_str());
            }else{evidence.complete=true;evidence.bytes=(uint64_t)info.st_size;}
        }
    }
    const char* entry=nvbit_get_func_name(ctx,function,true);
    const std::string api_return=evidence.dump_api_called?(evidence.dump_api_return?"true":"false"):"null";
    if(!evidence.complete||!entry||!entry[0])return "{\"complete\":false,\"reason\":"+quote(evidence.reason.empty()?"entry_selector_missing":evidence.reason.c_str())+",\"dump_api_return\":"+api_return+"}";
    complete=true;
    return "{\"complete\":true,\"module_id\":"+std::to_string(evidence.id)+",\"module_bytes\":"+std::to_string(evidence.bytes)+",\"dump_api_return\":"+api_return+",\"entry_selector\":"+quote(entry)+"}";
}
static std::string launch_attributes(const CUlaunchConfig* config){
    // Attribute values are request-local launch configuration, never pointer
    // identity.  Only fixed-size, pointer-free values are serialised.  Event,
    // graph-node and access-window attributes stay explicitly opaque rather
    // than smuggling process handles into a cross-turn configuration key.
    if(!config)return "{\"complete\":false,\"reason\":\"missing_config\"}";
    if(!config->numAttrs)return "{\"complete\":true,\"values\":[]}";
    if(!config->attrs||config->numAttrs>32)return "{\"complete\":false,\"reason\":\"invalid_or_excessive_attribute_count\"}";
    std::vector<std::string> values;std::vector<unsigned> unsupported;std::vector<unsigned> duplicate;
    std::unordered_set<unsigned> ids;
    for(unsigned i=0;i<config->numAttrs;++i){
        const CUlaunchAttribute& attribute=config->attrs[i];const unsigned id=(unsigned)attribute.id;
        if(!ids.insert(id).second){duplicate.push_back(id);continue;}
        switch(attribute.id){
            case CU_LAUNCH_ATTRIBUTE_IGNORE:values.push_back("{\"id\":\"ignore\",\"value\":true}");break;
            case CU_LAUNCH_ATTRIBUTE_COOPERATIVE:values.push_back("{\"id\":\"cooperative\",\"value\":"+std::to_string(attribute.value.cooperative)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY:values.push_back("{\"id\":\"synchronization_policy\",\"value\":"+std::to_string((int)attribute.value.syncPolicy)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:values.push_back("{\"id\":\"cluster_dimension\",\"value\":["+std::to_string(attribute.value.clusterDim.x)+","+std::to_string(attribute.value.clusterDim.y)+","+std::to_string(attribute.value.clusterDim.z)+"]}");break;
            case CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:values.push_back("{\"id\":\"cluster_scheduling_policy_preference\",\"value\":"+std::to_string((int)attribute.value.clusterSchedulingPolicyPreference)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:values.push_back("{\"id\":\"programmatic_stream_serialization\",\"value\":"+std::to_string(attribute.value.programmaticStreamSerializationAllowed)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_PRIORITY:values.push_back("{\"id\":\"priority\",\"value\":"+std::to_string(attribute.value.priority)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:values.push_back("{\"id\":\"mem_sync_domain_map\",\"value\":["+std::to_string((unsigned)attribute.value.memSyncDomainMap.default_)+","+std::to_string((unsigned)attribute.value.memSyncDomainMap.remote)+"]}");break;
            case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:values.push_back("{\"id\":\"mem_sync_domain\",\"value\":"+std::to_string((int)attribute.value.memSyncDomain)+"}");break;
            case CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION:values.push_back("{\"id\":\"preferred_cluster_dimension\",\"value\":["+std::to_string(attribute.value.preferredClusterDim.x)+","+std::to_string(attribute.value.preferredClusterDim.y)+","+std::to_string(attribute.value.preferredClusterDim.z)+"]}");break;
            case CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:values.push_back("{\"id\":\"preferred_shared_memory_carveout\",\"value\":"+std::to_string(attribute.value.sharedMemCarveout)+"}");break;
            default:unsupported.push_back(id);break;
        }
    }
    std::sort(values.begin(),values.end());std::sort(unsupported.begin(),unsupported.end());std::sort(duplicate.begin(),duplicate.end());
    std::string result="{\"complete\":"+std::string(unsupported.empty()&&duplicate.empty()?"true":"false")+",\"values\":[";
    for(size_t i=0;i<values.size();++i)result+=(i?",":"")+values[i];
    result+="]";
    if(!unsupported.empty()){result+=",\"unsupported_ids\":[";for(size_t i=0;i<unsupported.size();++i)result+=(i?",":"")+std::to_string(unsupported[i]);result+="]";}
    if(!duplicate.empty()){result+=",\"duplicate_ids\":[";for(size_t i=0;i<duplicate.size();++i)result+=(i?",":"")+std::to_string(duplicate[i]);result+="]";}
    return result+"}";
}
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
    max_library_module_dumps=env_int("RUNTIME_TRACE_MAX_LIBRARY_MODULE_DUMPS",16);
    max_library_module_bytes=env_int("RUNTIME_TRACE_MAX_LIBRARY_MODULE_BYTES",64*1024*1024);
    if(max_library_module_dumps<1||max_library_module_dumps>256)fail("library module dump count out of range");
    if(max_library_module_bytes<1||max_library_module_bytes>512*1024*1024)fail("library module dump byte budget out of range");
    int total=env_int("RUNTIME_TRACE_TOTAL_RECORDS",-1);if(total>=0)remaining_records=total;
    const char* policy=getenv("RUNTIME_TRACE_MEMORY_POLICY");memory_enabled=!policy||strcmp(policy,"interfaces");
    skip_vendor=env_int("RUNTIME_TRACE_SKIP_VENDOR_MEMORY",0);
    const char* format=getenv("RUNTIME_TRACE_RECORD_FORMAT");
    if(format&&strcmp(format,"runs")&&strcmp(format,"regions"))fail("unknown record format");
    aggregate_regions=format&&!strcmp(format,"regions");
    const char* detail=getenv("RUNTIME_TRACE_ORDERED_LAUNCHES");
    if(detail&&*detail){
        if(!aggregate_regions)fail("ordered launch selection requires regions format");
        for(const char* at=detail;*at;){
            if(*at<'0'||*at>'9')fail("invalid ordered launch ID");
            char* end=nullptr;auto id=strtoull(at,&end,10);
            if(*end&&*end!=',')fail("invalid ordered launch separator");
            if(id>2047||!ordered_launches.insert(id).second)fail("invalid or repeated ordered launch ID");
            at=*end?end+1:end;if(*end&&! *at)fail("trailing ordered launch separator");
        }
        remaining_ordered_records=std::min(capacity,remaining_records);
    }
    if(capacity<1||capacity>16777216)fail("memory run capacity out of range");
    if(aggregate_regions&&(capacity&(capacity-1)))fail("region hash capacity must be a power of two");
    log_file=fopen((prefix+".jsonl").c_str(),"wx");if(!log_file)fail("trace output must be new");
    fprintf(log_file,"{\"type\":\"config\",\"schema\":\"coarse-memory-runs/v1\",\"nvbit\":\"%s\",\"cta_limit\":%d,\"capacity\":%lu,\"serialized_diagnostic\":true,\"max_launches\":%d,\"total_records\":%lld,\"memory_enabled\":%s,\"skip_vendor_memory\":%s,\"library_module_dump_limit\":%d,\"library_module_dump_byte_limit\":%d,\"counter_mode\":\"saturating_lower_bound\"}\n",NVBIT_VERSION,cta_limit,capacity,max_launches,(long long)remaining_records,memory_enabled?"true":"false",skip_vendor?"true":"false",max_library_module_dumps,max_library_module_bytes);flush();
    fprintf(log_file,"{\"type\":\"record_format\",\"format\":\"%s\",\"region_bytes\":%u,\"capacity_scope\":\"%s\",\"total_record_unit\":\"%s\",\"total_record_limit_applies\":true,\"region_table_bytes\":%lu}\n",aggregate_regions?"exact_byte_regions/v1":"ordered_memory_runs/v1",REGION_BYTES,aggregate_regions?"per_launch_tiles_and_dump_scratch":"per_launch_runs",aggregate_regions?"emitted_merged_intervals":"ordered_memory_runs",aggregate_regions?sizeof(RegionBuffer)+capacity*sizeof(MemoryTile):0);flush();
    if(!ordered_launches.empty()){fprintf(log_file,"{\"type\":\"ordered_selection\",\"launch_ids\":%s,\"ordered_record_budget\":%lu,\"budget_scope\":\"additional_whole_forward_ordered_records\"}\n",quote(detail).c_str(),remaining_ordered_records);flush();}
}
void nvbit_tool_init(CUcontext ctx){std::lock_guard<std::recursive_mutex> guard(lock);bool old=internal;internal=true;
    if(first_context&&first_context!=ctx)fail("multiple contexts unsupported");first_context=ctx;
    if(aggregate_regions){
        if(!region_buffer)CHECK(cuMemAllocManaged((CUdeviceptr*)&region_buffer,sizeof(RegionBuffer)+capacity*sizeof(MemoryTile),CU_MEM_ATTACH_GLOBAL));
        if(!ordered_launches.empty()&&!dispatch){
            CHECK(cuMemAllocManaged((CUdeviceptr*)&buffer,sizeof(TraceBuffer)+capacity*sizeof(MemoryRun),CU_MEM_ATTACH_GLOBAL));
            CHECK(cuMemAllocManaged((CUdeviceptr*)&dispatch,sizeof(CaptureDispatch),CU_MEM_ATTACH_GLOBAL));
            dispatch->runs=(uint64_t)buffer;dispatch->regions=(uint64_t)region_buffer;dispatch->ordered=0;
        }
    }else if(!buffer){CHECK(cuMemAllocManaged((CUdeviceptr*)&buffer,sizeof(TraceBuffer)+capacity*sizeof(MemoryRun),CU_MEM_ATTACH_GLOBAL));buffer->capacity=capacity;buffer->count=0;}internal=old;
}
static void inspect(CUcontext ctx,CUfunction function,bool insert){
    auto funcs=nvbit_get_related_functions(ctx,function);funcs.push_back(function);
    for(auto f:funcs){if(info.count(f)&&(!insert||instrumented.count(f)))continue;FunctionInfo fi;
        for(auto ins:nvbit_get_instrs(ctx,f)){
            for(const char* p=ins->getSass();*p;++p){fi.hash^=(unsigned char)*p;fi.hash*=1099511628211ULL;}
            auto space=ins->getMemorySpace();
            if(space==InstrType::MemorySpace::NONE||space==InstrType::MemorySpace::CONSTANT||space==InstrType::MemorySpace::SHARED||space==InstrType::MemorySpace::LOCAL)continue;
            int refs=0;for(int i=0;i<ins->getNumOperands();++i)refs+=ins->getOperand(i)->type==InstrType::OperandType::MREF;
            if(space!=InstrType::MemorySpace::GLOBAL||refs!=1||ins->getSize()<=0||(!ins->isLoad()&&!ins->isStore())){++fi.unsupported;continue;}
            if(!insert)continue;
            nvbit_insert_call(ins,dispatch?"trace_dispatch":aggregate_regions?"trace_regions":"trace_memory",IPOINT_BEFORE);nvbit_add_call_arg_guard_pred_val(ins);nvbit_add_call_arg_mref_addr64(ins,0);
            int mode=(ins->isLoad()?1:0)|(ins->isStore()?2:0);
            if(strstr(ins->getOpcode(),"ATOM")||strncmp(ins->getOpcode(),"RED",3)==0)mode=3;
            nvbit_add_call_arg_const_val32(ins,ins->getSize());nvbit_add_call_arg_const_val32(ins,mode);
            nvbit_add_call_arg_const_val64(ins,dispatch?(uint64_t)dispatch:aggregate_regions?(uint64_t)region_buffer:(uint64_t)buffer);nvbit_add_call_arg_const_val32(ins,(uint32_t)cta_limit);
        }info[f]=fi;if(insert)instrumented.insert(f);
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
static std::vector<MemoryRun> region_runs(){
    std::vector<MemoryRun> runs;
    auto append=[&](uint64_t address,unsigned bytes,unsigned mode){
        if(runs.size()>=capacity){region_buffer->overflow=1;return;}
        runs.push_back({address,0,bytes,1,mode,0});
    };
    for(uint64_t i=0;i<capacity;++i){
        if(region_buffer->overflow&&runs.size()>=capacity)break;
        const auto& tile=region_buffer->tiles[i];if(!tile.key)continue;
        uint64_t base=(tile.key-1)*REGION_BYTES;
        for(unsigned mode:{1u,2u}){
            const auto* words=mode==1?tile.reads:tile.writes;
            auto add=[&](uint64_t address,unsigned bytes){
                append(address,bytes,mode);
                if(mode==2&&(tile.flags&1))append(address,bytes,8);
                if(mode==2&&(tile.flags&2))append(address,bytes,4);
            };
            bool full=true;for(unsigned j=0;j<REGION_WORDS;++j)full=full&&words[j]==~0ULL;
            if(full){add(base,REGION_BYTES);continue;}
            for(unsigned j=0;j<REGION_WORDS;++j){
                auto bits=words[j];
                while(bits){
                    unsigned start=__builtin_ctzll(bits);auto shifted=bits>>start;
                    unsigned length=shifted==~0ULL?64:__builtin_ctzll(~shifted);
                    add(base+j*64+start,length);
                    auto mask=length==64?~0ULL:((1ULL<<length)-1)<<start;
                    bits&=~mask;
                }
            }
        }
    }
    std::sort(runs.begin(),runs.end(),[](const auto& a,const auto& b){return a.mode!=b.mode?a.mode<b.mode:a.address<b.address;});
    size_t n=0;
    for(const auto& run:runs){
        if(n&&runs[n-1].mode==run.mode&&run.address<=runs[n-1].address+runs[n-1].bytes&&
           std::max(run.address+run.bytes,runs[n-1].address+runs[n-1].bytes)-runs[n-1].address<=UINT32_MAX){
            auto& last=runs[n-1];last.bytes=std::max(last.address+last.bytes,run.address+run.bytes)-last.address;
        }else runs[n++]=run;
    }
    runs.resize(n);return runs;
}
void nvbit_at_cuda_event(CUcontext ctx,int exiting,nvbit_api_cuda_t cbid,const char* name,void* params,CUresult* status){
    if(internal)return;std::lock_guard<std::recursive_mutex> guard(lock);internal=true;
    bool ordinary=cbid==API_CUDA_cuLaunchKernel||cbid==API_CUDA_cuLaunchKernel_ptsz;
    bool extended=cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz;
    if(!ordinary&&!extended){
        // Invalidate exactly after a successful unload.  The next module
        // assigned this driver-handle value receives a fresh, monotonic dump
        // path and cannot inherit its predecessor's cubin identity.
        if(exiting&&cbid==API_CUDA_cuModuleUnload&&*status==CUDA_SUCCESS){
            auto* p=(cuModuleUnload_params*)params;library_modules.erase(p->hmod);
        }
        if(exiting&&cbid==API_CUDA_cuLibraryUnload&&*status==CUDA_SUCCESS){
            // CUkernel/library ownership may release one or more CUmodules;
            // without a complete library->module map, invalidate all cached
            // handles.  IDs remain monotonic, so no old dump path is reused.
            library_modules.clear();
        }
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
    CUfunction f;CUstream stream;void** arguments;unsigned gx,gy,gz,bx,by,bz,shared;const CUlaunchConfig* extended_config=nullptr;
    if(ordinary){auto* p=(cuLaunchKernel_params*)params;f=p->f;stream=p->hStream;arguments=p->kernelParams;gx=p->gridDimX;gy=p->gridDimY;gz=p->gridDimZ;bx=p->blockDimX;by=p->blockDimY;bz=p->blockDimZ;shared=p->sharedMemBytes;}
    else{auto* p=(cuLaunchKernelEx_params*)params;f=p->f;arguments=p->kernelParams;auto* c=p->config;extended_config=c;stream=c->hStream;gx=c->gridDimX;gy=c->gridDimY;gz=c->gridDimZ;bx=c->blockDimX;by=c->blockDimY;bz=c->blockDimZ;shared=c->sharedMemBytes;}
    if(!active){if(!exiting)nvbit_enable_instrumented(ctx,f,false);internal=false;return;}
    if(max_launches>=0&&launch_id>=uint64_t(max_launches)){
        if(!exiting){nvbit_enable_instrumented(ctx,f,false);if(launch_id==uint64_t(max_launches)){fprintf(log_file,"{\"type\":\"unsupported_launch\",\"seq\":%lu,\"name\":\"launch_budget_exhausted\"}\n",position++);flush();}}
        else ++launch_id;internal=false;return;
    }
    if(first_context&&first_context!=ctx)fail("multiple active CUDA contexts unsupported");
    if(launch_thread==std::thread::id())launch_thread=std::this_thread::get_id();if(launch_thread!=std::this_thread::get_id())fail("multiple host launch threads unsupported");
    if(!exiting){
        if(!buffer&&!region_buffer)nvbit_tool_init(ctx);internal=true;CHECK(cuCtxSynchronize());
        const char* symbol=nvbit_get_func_name(ctx,f);
        bool vendor=skip_vendor&&(strstr(symbol,"cudnn")||strstr(symbol,"cublas")||strstr(symbol,"cutlass::")||strstr(symbol,"xmma_")||strstr(symbol,"gemv2"));
        ordered_launch=ordered_launches.count(launch_id);
        auto available=ordered_launch?remaining_ordered_records:remaining_records;
        if(dispatch)dispatch->ordered=ordered_launch;
        skip_reason=library_parent>=0?"library_api_contract":!memory_enabled?"interface_only_policy":vendor?"vendor_memory_opaque":available==0?"record_budget_exhausted":"";
        selected=!skip_reason[0];
        // A supported cuBLAS parent already supplies its public operand
        // contract.  Its child uses a once-per-module cubin dump plus a
        // mangled binary entry selector, without instruction traversal or
        // memory callbacks; final use remains gated on API success.
        const bool library_child=library_parent>=0;
        bool library_binary_complete=false;
        std::string library_binary_summary=library_child?library_binary(ctx,f,library_binary_complete):"";
        bool inspect_implementation=!library_child&&!(skip_vendor&&vendor);
        if(inspect_implementation)inspect(ctx,f,selected);
        if(selected&&aggregate_regions&&!ordered_launch){
            // Like the ordered buffer header, initialize managed storage on
            // the host after synchronization. NVBit instruction inspection
            // does not guarantee a usable driver context for cuMemset here.
            std::memset(region_buffer,0,sizeof(RegionBuffer)+capacity*sizeof(MemoryTile));
            region_buffer->capacity=capacity;
        }else if(selected){buffer->count=0;buffer->capacity=std::min(capacity,available);buffer->truncated=0;}
        FunctionInfo aggregate=inspect_implementation?combined_info(ctx,f):FunctionInfo{};
        fprintf(log_file,"{\"type\":\"launch\",\"record_format\":\"%s\",\"seq\":%lu,\"id\":%lu,\"parent\":%lld,\"name\":%s,\"stream\":%lu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared\":%u,\"memory_traced\":%s,\"implementation\":\"%016lx\",\"unsupported_memory_instructions\":%d,\"arguments\":[",aggregate_regions&&!ordered_launch?"exact_byte_regions/v1":"ordered_memory_runs/v1",position++,launch_id,library_parent,quote(nvbit_get_func_name(ctx,f)).c_str(),(uint64_t)stream,gx,gy,gz,bx,by,bz,shared,selected?"true":"false",selected?aggregate.hash:0,selected?aggregate.unsupported:0);
        bool inspect_arguments=!(skip_vendor&&(vendor||library_parent>=0));
        auto sizes=inspect_arguments?nvbit_get_kernel_argument_sizes(ctx,f):std::vector<int>{};
        if(arguments)for(size_t i=0;i<sizes.size();++i){uint64_t value=0;if(sizes[i]<=8)memcpy(&value,arguments[i],sizes[i]);fprintf(log_file,"%s{\"index\":%lu,\"bytes\":%d,\"bits\":%lu}",i?",":"",i,sizes[i],value);}
        unsigned num_attrs=ordinary?0:extended_config->numAttrs;
        std::string attribute_summary=ordinary?"{\"complete\":true,\"values\":[]}":launch_attributes(extended_config);
        fprintf(log_file,"],\"arguments_present\":%s,\"launch_api\":\"%s\",\"launch_num_attrs\":%u,\"launch_attributes\":%s,\"memory_skip_reason\":%s,\"implementation_version\":\"%016lx\",\"implementation_inspected\":%s%s%s}\n",inspect_arguments&&(arguments||sizes.empty())?"true":"false",ordinary?"cuLaunchKernel":"cuLaunchKernelEx",num_attrs,attribute_summary.c_str(),quote(skip_reason).c_str(),aggregate.hash,(library_child?library_binary_complete:inspect_implementation)?"true":"false",library_child?",\"library_binary\":" : "",library_child?library_binary_summary.c_str():"");nvbit_enable_instrumented(ctx,f,selected);flush();
    }else{
        CHECK(*status);CHECK(cuCtxSynchronize());
        std::vector<MemoryRun> summarized;
        uint64_t kept=0,dropped=0;bool exact=true;const MemoryRun* data=nullptr;
        if(selected&&aggregate_regions&&!ordered_launch){
            summarized=region_runs();kept=std::min(uint64_t(summarized.size()),remaining_records);data=summarized.data();
            if(kept<summarized.size())region_buffer->overflow=1;
            remaining_records-=kept;
            dropped=region_buffer->overflow?1:0;exact=!region_buffer->overflow;
        }else if(selected){
            kept=std::min((uint64_t)buffer->count,(uint64_t)buffer->capacity);
            if(ordered_launch)remaining_ordered_records-=kept;else remaining_records-=kept;data=buffer->runs;
            dropped=(uint64_t)buffer->count-kept+buffer->truncated;exact=!buffer->truncated;
        }
        if(selected){std::string path=prefix+".launch"+std::to_string(launch_id)+".bin";FILE* f=fopen(path.c_str(),"wbx");if(!f)fail("binary must be new");if(fwrite(data,sizeof(MemoryRun),kept,f)!=kept||fclose(f))fail("binary write failed");}
        fprintf(log_file,"{\"type\":\"complete\",\"seq\":%lu,\"id\":%lu,\"runs\":%lu,\"dropped\":%lu,\"drop_count_exact\":%s}\n",position++,launch_id,kept,dropped,exact?"true":"false");++launch_id;flush();
    }internal=false;
}
void nvbit_at_term(){if(log_file){fprintf(log_file,"{\"type\":\"end\",\"seq\":%lu,\"launches\":%lu}\n",position++,launch_id);flush();fclose(log_file);}}
