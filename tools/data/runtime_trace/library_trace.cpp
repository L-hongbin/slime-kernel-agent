// Runtime cuBLAS contracts; library children remain visible as kernel launches.
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <string>
#include <mutex>
#include <unordered_map>
extern "C" int coarse_active();
extern "C" long long coarse_library_begin(const char*,const char*,uint64_t);
extern "C" void coarse_library_end(long long,int);
struct Workspace {size_t bytes=0;bool known=false;bool custom=false;bool explicitly_set=false;};
static std::mutex state_lock;
static std::unordered_map<cublasHandle_t,Workspace> workspace;
template<class Fn> static Fn real(const char* name){
    void* p=dlsym(RTLD_NEXT,name);
    if(!p){void* lib=dlopen("libcublas.so.12",RTLD_LAZY|RTLD_NOLOAD);if(lib){p=dlsym(lib,name);dlclose(lib);}}
    if(!p){fprintf(stderr,"COARSE_MISSING_LIBRARY_SYMBOL %s\n",name);abort();}
    return reinterpret_cast<Fn>(p);
}
static std::string number(double v){char text[80];if(!std::isfinite(v))return "null";snprintf(text,sizeof(text),"%.17g",v);return text;}
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* h){static auto fn=real<decltype(&cublasCreate_v2)>("cublasCreate_v2");auto status=fn(h);if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(state_lock);workspace[*h]={0,true,false};}return status;}
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t h){static auto fn=real<decltype(&cublasDestroy_v2)>("cublasDestroy_v2");auto status=fn(h);if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(state_lock);workspace.erase(h);}return status;}
extern "C" cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h,void* p,size_t bytes){static auto fn=real<decltype(&cublasSetWorkspace_v2)>("cublasSetWorkspace_v2");auto status=fn(h,p,bytes);if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(state_lock);workspace[h]={bytes,true,p!=nullptr,true};}return status;}
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t h,cudaStream_t stream){static auto fn=real<decltype(&cublasSetStream_v2)>("cublasSetStream_v2");auto status=fn(h,stream);if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(state_lock);workspace[h]={0,true,false};}return status;}
static long long begin(const char* api,cublasHandle_t handle,int ta,int tb,int m,int n,int k,
    const void* alpha,const void* a,int at,int lda,const void* b,int bt,int ldb,
    const void* beta,void* c,int ct,int ldc,int compute,int algorithm){
    if(!coarse_active())return -1;
    cudaStream_t stream=nullptr;cublasPointerMode_t pointer=CUBLAS_POINTER_MODE_DEVICE;cublasMath_t math=CUBLAS_DEFAULT_MATH;
    auto gs=real<decltype(&cublasGetStream_v2)>("cublasGetStream_v2")(handle,&stream);
    auto gp=real<decltype(&cublasGetPointerMode_v2)>("cublasGetPointerMode_v2")(handle,&pointer);
    auto gm=real<decltype(&cublasGetMathMode)>("cublasGetMathMode")(handle,&math);
    bool host=gp==CUBLAS_STATUS_SUCCESS&&pointer==CUBLAS_POINTER_MODE_HOST;
    bool f32=compute==CUBLAS_COMPUTE_32F||compute==CUBLAS_COMPUTE_32F_FAST_TF32||compute==CUBLAS_COMPUTE_32F_PEDANTIC||compute==CUBLAS_COMPUTE_32F_FAST_16F||compute==CUBLAS_COMPUTE_32F_FAST_16BF;
    bool f64=compute==CUBLAS_COMPUTE_64F||compute==CUBLAS_COMPUTE_64F_PEDANTIC;
    std::string av="null",bv="null";
    if(host&&alpha&&beta){if(f32){av=number(*(const float*)alpha);bv=number(*(const float*)beta);}else if(f64){av=number(*(const double*)alpha);bv=number(*(const double*)beta);}}
    char data[1800];snprintf(data,sizeof(data),"{\"ta\":%d,\"tb\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"a\":%lu,\"b\":%lu,\"c\":%lu,\"at\":%d,\"bt\":%d,\"ct\":%d,\"lda\":%d,\"ldb\":%d,\"ldc\":%d,\"alpha\":%s,\"beta\":%s,\"compute\":%d,\"algorithm\":%d,\"math_mode\":%d,\"pointer_mode\":%d,\"state_query_ok\":%s}",ta,tb,m,n,k,(uint64_t)a,(uint64_t)b,(uint64_t)c,at,bt,ct,lda,ldb,ldc,av.c_str(),bv.c_str(),compute,algorithm,(int)math,(int)pointer,(gs==CUBLAS_STATUS_SUCCESS&&gm==CUBLAS_STATUS_SUCCESS&&gp==CUBLAS_STATUS_SUCCESS)?"true":"false");
    Workspace ws;{std::lock_guard<std::mutex> guard(state_lock);ws=workspace[handle];}
    int version=0;auto version_status=real<decltype(&cublasGetVersion_v2)>("cublasGetVersion_v2")(handle,&version);
    std::string attributes=data;attributes.pop_back();attributes+=",\"workspace_known\":";attributes+=ws.known?"true":"false";
    attributes+=",\"workspace_custom\":";attributes+=ws.custom?"true":"false";
    const char* mode=!ws.known?"unknown":!ws.explicitly_set?"default_pool":ws.bytes==0?"default_pool_disabled":ws.custom?"custom":"unknown";
    attributes+=",\"workspace_mode\":\""+std::string(mode)+"\",\"workspace_bytes\":"+std::to_string(ws.bytes)+",\"library_version\":"+(version_status==CUBLAS_STATUS_SUCCESS?std::to_string(version):"null")+"}";
    return coarse_library_begin(api,attributes.c_str(),(uint64_t)stream);
}
extern "C" cublasStatus_t cublasSgemm_v2(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const float* alpha,const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc){
    static auto fn=real<decltype(&cublasSgemm_v2)>("cublasSgemm_v2");
    auto id=begin("cublasSgemm_v2",h,ta,tb,m,n,k,alpha,a,CUDA_R_32F,lda,b,CUDA_R_32F,ldb,beta,c,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,-1);
    auto status=fn(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);coarse_library_end(id,status);return status;
}
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const void* alpha,const void* a,cudaDataType at,int lda,const void* b,cudaDataType bt,int ldb,
    const void* beta,void* c,cudaDataType ct,int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo){
    using Fn=cublasStatus_t(*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const void*,const void*,cudaDataType,int,const void*,cudaDataType,int,const void*,void*,cudaDataType,int,cublasComputeType_t,cublasGemmAlgo_t);
    static auto fn=real<Fn>("cublasGemmEx");
    auto id=begin("cublasGemmEx",h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,compute,algo);
    auto status=fn(h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,compute,algo);coarse_library_end(id,status);return status;
}
