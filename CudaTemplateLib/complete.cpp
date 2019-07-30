//file.h :

template<typename T>
__global__ void Add1(T a, T b);

template<typename T>
__host__ __device__ void Add2(T a, T b);

//file_template.h :

template<typename T>
__global__ void Add1(T a, T b)
{
  a = a+b;
}

template<typename T>
__host__ __device__ T Add2(T a, T b)
{
  return a+b;
}

//file.cu :

#include "test_template.h"

template float Add2<float>(float a, float b);
template int Add2<int>(int a, int b);

__host__ void dummyHost(void)
{
  //Add<float>(1,1);
  Add1<float><<<1,1,0,0>>>(1.0,1.0);
  Add1<int><<<1,1,0,0>>>(1,1);
}

__device__ void dummyDevice(void)
{
  Add2<float>(1.0, 1.0);
  Add2<int>(1,1);
}