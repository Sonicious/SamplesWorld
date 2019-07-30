//file.h :

template<typename T>
__global__ void Add1(T a, T b);

template<typename T>
__host__ __device__ void Add2(T a, T b);