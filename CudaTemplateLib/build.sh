nvcc --shared -o libtest.so test.cu --compiler-options '-fPIC'
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
nvcc -L. -ltest -dc tt.cu -o tester