#include <dlfcn.h>
#include <string.h>
#include <iostream>
#include <map>
#include <string>

#include <cuda.h>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x

static void* realCuGetProcAddress;
void* libcudaHandle = dlopen("libcuda.so", RTLD_LAZY);
void* libdlHandle = dlopen("libdl.so", RTLD_LAZY);

static std::map<CUfunction, std::string> cu_func_map;

CUresult cuGetProcAddress_custom(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
CUresult cuLaunchKernel_custom(CUfunction f,
                               unsigned int gridDimX, 
                               unsigned int gridDimY, 
                               unsigned int gridDimZ, 
                               unsigned int blockDimX, 
                               unsigned int blockDimY, 
                               unsigned int blockDimZ, 
                               unsigned int sharedMemBytes, 
                               CUstream hStream, 
                               void** kernelParams, 
                               void** extra);
CUresult cuModuleGetFunction_custom(CUfunction* hfunc, CUmodule hmod, const char* name);

static void *real_dlsym(void *handle, const char *symbol) {
    typedef void *(*fnDlsym)(void *, const char *);
    static fnDlsym internal_dlsym = (fnDlsym)dlvsym(libdlHandle, "dlsym", "GLIBC_2.2.5");

    return (*internal_dlsym)(handle, symbol);
}

void* dlsym(void *handle, const char *symbol) 
{
    if (strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress_custom);
    }

    return real_dlsym(handle, symbol);
}

CUresult cuGetProcAddress_custom(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    typedef decltype(&cuGetProcAddress) funcType;
    funcType realFn = NULL;
    if (!realCuGetProcAddress) {
        realFn = (funcType)real_dlsym(libcudaHandle, STRINGIFY(cuGetProcAddress));
    } else {
        realFn = (funcType)realCuGetProcAddress;
    }
    CUresult result = realFn(symbol, pfn, cudaVersion, flags);

    if (result == CUDA_SUCCESS) {
        if (strcmp(symbol, STRINGIFY_AUX(cuGetProcAddress)) == 0) {
            realCuGetProcAddress = *pfn;
            *pfn = (void*)(&cuGetProcAddress_custom);
        } else if (strcmp(symbol, STRINGIFY_AUX(cuLaunchKernel)) == 0) {
            *pfn = (void*)(&cuLaunchKernel_custom);
        } else if (strcmp(symbol, STRINGIFY_AUX(cuModuleGetFunction)) == 0) {
            *pfn = (void*)(&cuModuleGetFunction_custom);
        }
    }

    return result;
}

CUresult cuLaunchKernel_custom(CUfunction f,
                               unsigned int gridDimX, 
                               unsigned int gridDimY, 
                               unsigned int gridDimZ, 
                               unsigned int blockDimX, 
                               unsigned int blockDimY, 
                               unsigned int blockDimZ, 
                               unsigned int sharedMemBytes, 
                               CUstream hStream, 
                               void** kernelParams,
                               void** extra) {
    if (cu_func_map.find(f) != cu_func_map.end()) {
        std::string cu_func_name = cu_func_map.find(f)->second;
        
        if (cu_func_name.find("cudnn") != std::string::npos || cu_func_name.find("conv") != std::string::npos || cu_func_name.find("elementwise") != std::string::npos) {
            std::cout << "Function Name : " << cu_func_name << std::endl;

            // std::cout << "BEFORE" << std::endl;
            std::cout << "    Grid Size : " << gridDimX << " " << gridDimY << " " << gridDimZ << std::endl;
            std::cout << "    Block Size : " << blockDimX << " " << blockDimY << " " << blockDimZ << std::endl;

            // if (gridDimX % 32 == 0) {
            //     blockDimX = blockDimX * (gridDimX / 32);
            //     gridDimX = 32;
            // } else if (blockDimX % 32 == 0) {
            //     gridDimX = gridDimX * (blockDimX / 32);
            //     blockDimX = 32;
            // }

            // std::cout << "AFTER" << std::endl;
            // std::cout << "    Grid Size : " << gridDimX << " " << gridDimY << " " << gridDimZ << std::endl;
            // std::cout << "    Block Size : " << blockDimX << " " << blockDimY << " " << blockDimZ << std::endl;
            std::cout << std::endl;
        }
    }

    typedef decltype(&cuLaunchKernel) funcType;
    funcType realFn = (funcType)real_dlsym(libcudaHandle, STRINGIFY(cuLaunchKernel));
    CUresult result = realFn(f, 
                             gridDimX, 
                             gridDimY, 
                             gridDimZ, 
                             blockDimX, 
                             blockDimY, 
                             blockDimZ, 
                             sharedMemBytes, 
                             hStream, 
                             kernelParams, 
                             extra);

    return result;
}

CUresult cuModuleGetFunction_custom(CUfunction* hfunc, CUmodule hmod, const char* name) {
    typedef decltype(&cuModuleGetFunction) funcType;
    funcType realFn = (funcType)real_dlsym(libcudaHandle, STRINGIFY(cuModuleGetFunction));
    CUresult result = realFn(hfunc, hmod, name);

    if (result == CUDA_SUCCESS) {
        cu_func_map.insert({*hfunc, std::string(name)});
    }

    return result;
}