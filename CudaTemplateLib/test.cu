//file.cu :

#include "test_template.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//template float Add2<float>(float a, float b);
//template int Add2<int>(int a, int b);

__host__ void dummyHost(void)
{
  Add1<float><<<1,1,0,0>>>(1.0,1.0);
  Add1<int><<<1,1,0,0>>>(1,1);
}

__device__ void dummyDevice(void)
{
  Add2<float>(1.0, 1.0);
  Add2<float>(1,1);
  Add2<int>(1.0,1.0);
  Add2<int>(1,1);
}