//#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <iostream>

//#include "ErrorHelper.hpp"

int main(int argc, char **argv)
{
    // 
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cout << "no OpenCL Device found\n";
        return EXIT_FAILURE;
    }

    // Print number of platforms and list of platforms
    std::cout << "Found " << platforms.size() << " OpenCL Platforms:" << std::endl;
    for (auto &platform : platforms)
    {
        std::string platformVendor;
        std::string platformName;
        std::string platformVersion;
        platform.getInfo(CL_PLATFORM_VENDOR, &platformVendor);
        platform.getInfo(CL_PLATFORM_NAME, &platformName);
        platform.getInfo(CL_PLATFORM_VERSION, &platformVersion);
        std::cout << "Platform 0: " << platformVendor
            << " : " << platformName 
            << " : " << platformVersion 
            << std::endl;
    }

    return EXIT_SUCCESS;
}