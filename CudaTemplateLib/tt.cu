#include "test_template.h"
#include <cstdio>

#include <cuda_runtime.h>

__global__ void deviceTest(void)
{
  Add2(1.0,1.0);
  Add2(1,1);
}

__host__ void hostTest(void)
{
  Add2(1.0,1.0);
  Add2(1,1);
}

int main(void)
{
  float *x, *y;
  int *a, *b;
  cudaMallocManaged(&x, 1*sizeof(float));
  cudaMallocManaged(&y, 1*sizeof(float));
  cudaMallocManaged(&a, 1*sizeof(int));
  cudaMallocManaged(&b, 1*sizeof(int));

  *x = 1.0f;
  *y = 1.0f;
  *a = 1;
  *b = 1;

  Add1<float><<<1,1>>>(*x, *y);
  Add1<int><<<1,1>>>(*a, *b);

  deviceTest<<<1,1>>>();
}