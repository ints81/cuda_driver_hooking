#include <dlfcn.h>
#include <string.h>
#include <iostream>

#include <cuda.h>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x

static void* realCuGetProcAddress;
void* libcudaHandle = dlopen("libcuda.so", RTLD_LAZY);
void* libdlHandle = dlopen("libdl.so", RTLD_LAZY);

CUresult cuGetProcAddress_custom(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
CUresult cuLaunchKernel_custom(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

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

    if (strcmp(symbol, STRINGIFY_AUX(cuGetProcAddress)) == 0) {
        realCuGetProcAddress = *pfn;
        *pfn = (void*)(&cuGetProcAddress_custom);
    } else if (strcmp(symbol, STRINGIFY_AUX(cuLaunchKernel)) == 0) {
        *pfn = (void*)(&cuLaunchKernel_custom);
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
    std::cout << "############################# BEFORE ##############################" << std::endl;
    std::cout << "Grid Size : " << gridDimX << " " << gridDimY << " " << gridDimZ << std::endl;
    std::cout << "Block Size : " << blockDimX << " " << blockDimY << " " << blockDimZ << std::endl;

    if (gridDimX % 32 == 0) {
        blockDimX = blockDimX * (gridDimX / 32);
        gridDimX = 32;
    } else if (blockDimX % 32 == 0) {
        gridDimX = gridDimX * (blockDimX / 32);
        blockDimX = 32;
    }

    std::cout << "############################# AFTER ##############################" << std::endl;
    std::cout << "Grid Size : " << gridDimX << " " << gridDimY << " " << gridDimZ << std::endl;
    std::cout << "Block Size : " << blockDimX << " " << blockDimY << " " << blockDimZ << std::endl;
    std::cout << std::endl;

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