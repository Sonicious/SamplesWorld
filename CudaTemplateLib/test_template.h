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