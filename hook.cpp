#include <dlfcn.h>
#include <string.h>
#include <iostream>

#include <cuda.h>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x

static void* realCuGetProcAddress;
void* libcudaHandle = dlopen("libcuda.so", RTLD_LAZY);

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

static void *real_dlsym(void *handle, const char *symbol) {
    typedef void *(*fnDlsym)(void *, const char *);
    static fnDlsym internal_dlsym = (fnDlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");

    return (*internal_dlsym)(handle, symbol);
}

void* dlsym(void *handle, const char *symbol) 
{
    if (strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
    }

    return real_dlsym(handle, symbol);
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    printf("[DEBUG] This is intercepted cuGetProcAddress!!!\n");
    printf("[DEBUG] Symbol : %s\n", symbol);

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
        *pfn = (void*)(&cuGetProcAddress);
    } else {
        printf("[DEBUG] This is other CUDA Driver api function.\n");
    }

    return result;
}