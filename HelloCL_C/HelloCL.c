// Most devices still use OpenCL 1.2:
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

void checkCL(cl_int error)
{
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL error: %i\n", error);
        exit(error);
    }
}

void printPlatformInfo(cl_platform_id platformId)
{
    char buffer[1024];
    checkCL(clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL));
    printf("\t%s", buffer);
    checkCL(clGetPlatformInfo(platformId, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
    printf(" : %s\n", buffer);
    checkCL(clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL));
    printf("\t%s\n", buffer);
    checkCL(clGetPlatformInfo(platformId, CL_PLATFORM_PROFILE, sizeof(buffer), buffer, NULL));
    printf("\t%s\n", buffer);
    checkCL(clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, sizeof(buffer), buffer, NULL));
    printf("\t%s\n", buffer);
}

void printDeviceInfo(cl_device_id deviceId)
{
    char buffer[1024];
    cl_device_type deviceType;
    checkCL(clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
    printf("\t%s", buffer);
    checkCL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
    printf(" : %s\n", buffer);
    checkCL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL));
    switch(deviceType)
    {
        case CL_DEVICE_TYPE_CPU: printf("\tCPU Device\n"); break;
        case CL_DEVICE_TYPE_GPU: printf("\tGPU Device\n"); break;
        case CL_DEVICE_TYPE_ACCELERATOR: printf("\tAccelerator Device\n"); break;
        case CL_DEVICE_TYPE_DEFAULT: printf("\tDefault Device\n"); break;
    }
    checkCL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
    printf("\t%s\n", buffer);
    checkCL(clGetDeviceInfo(deviceId, CL_DEVICE_PROFILE, sizeof(buffer), buffer, NULL));
    printf("\t%s\n", buffer);
}
 
int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *hostA = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *hostB = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *hostC = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++)
    {
        hostA[i] = i;
        hostB[i] = LIST_SIZE - i;
        hostC[i] = 0;
    }
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *sourceStr;
    size_t sourceSize;
    fp = fopen("kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(EXIT_FAILURE);
    }
    sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread( sourceStr, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Get number of platforms and according information
    cl_uint numPlatforms;
    checkCL(clGetPlatformIDs(0, NULL, &numPlatforms));
    if (numPlatforms != 0)
    {
        printf("Found %i OpenCL Platform(s)\n", numPlatforms);
    }
    else
    {
        fprintf(stderr, "no OpenCL Platforms available.\n");
        exit(EXIT_FAILURE);
    }
    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    checkCL(clGetPlatformIDs(numPlatforms, platforms, NULL));
    for (int i = 0; i < numPlatforms; ++i)
    {
        printf("Platform: %i\n", i);
        printPlatformInfo(platforms[i]);
    }
    printf("Using platform 0 for further program\n");

    // Get Devices and according information:
    cl_uint numDevices;
    checkCL(clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
    cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    printf("Found %i device(s) on platform 0:\n", numDevices);
    checkCL(clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL));
    for (int i = 0; i < numDevices; ++i)
    {
        printf("Device: %i\n", i);
        printDeviceInfo(devices[i]);
    }
    printf("Using platform 0, device 0 for further program\n");
 
    // Create an OpenCL context
    cl_int error = CL_SUCCESS;
    cl_platform_id myPlatform = platforms[0];
    cl_device_id myDevice = devices[0];
    cl_context_properties contextProps[3];
    contextProps[0] = (cl_context_properties) CL_CONTEXT_PLATFORM;
    contextProps[1] = (cl_context_properties) myPlatform;
    contextProps[2] = (cl_context_properties) 0; // last element must be 0
    // cfreate context with first device from first platform, no callback
    cl_context context = clCreateContext( contextProps, 1, &myDevice, NULL, NULL, &error);
    checkCL(error);
    printf("Created OpenCL Context\n");
 
    // Create a command queue
    // possible properties (3rd agument):
    //  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    //  CL_QUEUE_PROFILING_ENABLE
    cl_command_queue commandQueue = clCreateCommandQueue(context, myDevice, 0, &error);
    checkCL(error);
    printf("Created OpenCL Command Queue\n");
 
    // Create memory buffers on the device for each vector 
    cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &error);
    cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &error);
    cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &error);
 
    // Copy the lists A and B to their respective memory buffers
    // copy operation is blocking, without offset and doesn't depend on Events
    checkCL(clEnqueueWriteBuffer(commandQueue, deviceA, CL_TRUE, 0, LIST_SIZE * sizeof(int), hostA, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(commandQueue, deviceB, CL_TRUE, 0, LIST_SIZE * sizeof(int), hostB, 0, NULL, NULL));
    printf("Memory written to Device\n");
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceStr, (const size_t *)&sourceSize, &error);
    checkCL(error);
    checkCL(clBuildProgram(program, 1, &myDevice, NULL, NULL, NULL));
    cl_build_status buildStatus;
    char *buildLog;
    size_t buildLogSize;
    checkCL(clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_STATUS, 0, &buildStatus, NULL));
    if(buildStatus != CL_BUILD_SUCCESS)
    {
        printf("Problem: %i\n", buildStatus);
        checkCL(clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize));
        printf("Build Log:\n");
        buildLog = (char*)malloc(buildLogSize);
        checkCL(clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL));
        printf("%s\n", buildLog);
        exit(EXIT_FAILURE);
    }
    printf("Program built!\n");
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &error);
    checkCL(error);
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceA));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceB));
    checkCL(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&deviceC));
    printf("Kernel is created and arguments are set\n");
 
    // Execute the OpenCL kernel on the list
    size_t globalItemSize = LIST_SIZE; // Process the entire lists
    size_t localItemSize = 64; // Device dependent
    // Enqueue the kernel without events:
    checkCL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL));
    printf("kernel started\n");
 
    // Read the memory buffer C on the device to the local variable C
    checkCL(clEnqueueReadBuffer(commandQueue, deviceC, CL_TRUE, 0, LIST_SIZE * sizeof(int), hostC, 0, NULL, NULL));
    printf("Read from Buffer\n");
 
    // Display the result to the screen
    int correctResult = 1;
    for(i = 0; i < LIST_SIZE; i++)
    {
        if(hostC[i] != 1024) correctResult = 0;
    }
    if(correctResult)
    {
        printf("Result of Kernel is correct!\n");
    }
    else
    {
        printf("Result is wrong!\n");
    }

    // Clean up
    checkCL(clReleaseKernel(kernel));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseMemObject(deviceA));
    checkCL(clReleaseMemObject(deviceB));
    checkCL(clReleaseMemObject(deviceC));
    checkCL(clFlush(commandQueue));
    checkCL(clFinish(commandQueue));
    checkCL(clReleaseCommandQueue(commandQueue));
    checkCL(clReleaseContext(context));
    free(hostA);
    free(hostB);
    free(hostC);
    free(sourceStr);
    free(platforms);
    free(devices);
    // free(buildLog);
    return EXIT_SUCCESS;
}