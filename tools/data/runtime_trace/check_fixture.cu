// Executable calibration cases for the offline tracer; no task-specific rules
// are used by the extractor. Same inputs across implementation variants.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;} } while(0)
__global__ void add_one(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i]+1.0f;}
__device__ __forceinline__ float inc(float value){return value+1.0f;}
__global__ void renamed(const float* input,float* output,int size){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at<size)output[at]=inc(input[at]);}
__global__ void times_two(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i]*2.0f;}
__global__ void times_three(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i]*3.0f;}
__global__ void fused(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=(a[i]+1.0f)*3.0f;}
__global__ void transpose_shared(const float* a,float* b){__shared__ float tile[4][4];int x=threadIdx.x%4,y=threadIdx.x/4;tile[y][x]=a[y*4+x];__syncthreads();b[y*4+x]=tile[x][y];}
__global__ void race(float* b){b[0]=float(threadIdx.x);}
__global__ void fill_range(float* b,int begin,int count,float value){int i=threadIdx.x;if(i<count)b[begin+i]=value;}
__global__ void predicated_store(float* output){int i=threadIdx.x;float v=float(i+1);int even=i%2;asm volatile("{.reg .pred p; setp.eq.u32 p, %1, 0; @!p st.global.f32 [%0], %2;}"::"l"(output+i),"r"(even),"f"(v):"memory");}
int main(int argc,char**argv){int mode=argc>1?atoi(argv[1]):0;const int n=16;float a[n],out[n];for(int i=0;i<n;++i)a[i]=float(i);float *input,*tmp,*output;CUDA(cudaMalloc(&input,sizeof a));CUDA(cudaMalloc(&tmp,sizeof a));CUDA(cudaMalloc(&output,sizeof a));CUDA(cudaMemcpy(input,a,sizeof a,cudaMemcpyHostToDevice));
 printf("{\"allocations\":[{\"role\":\"input:0\",\"base\":%llu,\"bytes\":%zu},{\"role\":\"scratch:0\",\"base\":%llu,\"bytes\":%zu},{\"role\":\"output:0\",\"base\":%llu,\"bytes\":%zu}]}\n",(unsigned long long)input,sizeof a,(unsigned long long)tmp,sizeof a,(unsigned long long)output,sizeof a);
 if(mode==8 || mode==9){cudaStream_t s1,s2;cudaEvent_t event;CUDA(cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking));CUDA(cudaStreamCreateWithFlags(&s2,cudaStreamNonBlocking));CUDA(cudaEventCreate(&event));add_one<<<1,32,0,s1>>>(input,tmp,n);if(mode==8){CUDA(cudaEventRecord(event,s1));CUDA(cudaStreamWaitEvent(s2,event));}times_two<<<1,32,0,s2>>>(tmp,output,n);CUDA(cudaDeviceSynchronize());CUDA(cudaEventDestroy(event));CUDA(cudaStreamDestroy(s1));CUDA(cudaStreamDestroy(s2));}
 else if(mode==10){fill_range<<<1,32>>>(tmp,0,n,0.0f);fill_range<<<1,32>>>(tmp,4,4,1.0f);times_two<<<1,32>>>(tmp,output,n);}
 else if(mode==11){add_one<<<1,32>>>(input,input,n);times_two<<<1,32>>>(input,output,n);}
 else if(mode==7){CUDA(cudaMemset(output,0,sizeof a));predicated_store<<<1,16>>>(output);}
 else if(mode==4) fused<<<1,32>>>(input,output,n);
 else if(mode==5) transpose_shared<<<1,16>>>(input,output);
 else if(mode==6) race<<<1,32>>>(output);
 else {if(mode==1)renamed<<<1,32>>>(input,tmp,n);else add_one<<<1,32>>>(input,tmp,n);if(mode==2 || mode==3)times_three<<<1,32>>>(tmp,output,n);else times_two<<<1,32>>>(tmp,output,n);}
 CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(out,output,sizeof out,cudaMemcpyDeviceToHost));printf("{\"mode\":%d,\"output\":[",mode);for(int i=0;i<n;++i)printf("%s%.9g",i?",":"",out[i]);printf("]}\n");CUDA(cudaFree(input));CUDA(cudaFree(tmp));CUDA(cudaFree(output));return 0;}
