# CUDA Driver API Hooking

`LD_PRELOAD` 환경변수를 이용한 CUDA Driver API hooking 방법에 관한 repository이다.

다음과 같은 방법으로 빌드하고 실행한다.
```bash
# Make .so file
nvcc -shared -lcuda -ldl --compiler-options '-fPIC' hook.cpp -o hook.so

# Make executable file
nvcc -lcuda test.cu -o test
nvcc -lcuda -lcudnn -o conv_test conv_test.cu

# Execute with LD_PRELOAD
LD_PRELOAD=./hook.so ./test
LD_PRELOAD=./hook.so python torch_op_test.py
```