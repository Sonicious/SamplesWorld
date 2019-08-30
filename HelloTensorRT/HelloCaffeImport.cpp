#include <iostream>
#include <memory>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("[ERROR] %s\n", cudaGetErrorName(result));
        printf("[ERROR] %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void print_TensorRT_version()
{
    std::cout << "  TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << "."
              << NV_TENSORRT_BUILD << std::endl;
}

void print_CUDA_version()
{
  int version;
  checkCuda( cudaDriverGetVersion(&version) );
  std::cout << "  Cuda version: "
            << version/1000 << "."
            << (version%1000)/10
            << std::endl;
}

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) override
  {
    // suppress info-level messages
    //if (severity != Severity::kVERBOSE)
    std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

int main(int argc, char *argv[])
{

  // Version Information
  std::cout << "Hello TensorRT World"
            << std::endl;
  print_TensorRT_version();
  print_CUDA_version();

  // Create Inference Builder
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
  if(!builder)
  {
    return EXIT_FAILURE;
  }

  // Create Network
  nvinfer1::INetworkDefinition *network = builder->createNetwork();
  if(!network)
  {
    return EXIT_FAILURE;
  }

  size_t numberInputs = 7;

  // Add Layers to Network
  std::cout << "[LOG] " << "Build Network" <<std::endl;
  nvinfer1::ITensor *input = network->addInput("InputLayer", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3(numberInputs,1, 1));
  nvinfer1::IIdentityLayer *layer1 = network->addIdentity(*input);
  nvinfer1::ISoftMaxLayer *layer2 = network->addSoftMax(*input);
  nvinfer1::IIdentityLayer *output = network->addIdentity(*layer2->getOutput(0));
  output->getOutput(0)->setName("OutputLayer");
  network->markOutput(*output->getOutput(0));

  // Make Engine out of Network
  std::cout << "[LOG] " << "Build Cuda Engine" <<std::endl;
  nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);

  // Serialize the Engine
  std::cout << "[LOG] " << "Build Serialize Engine" <<std::endl;
  nvinfer1::IHostMemory *serializedEngine = engine->serialize();

  // Clean up a bit
  std::cout << "[LOG] " << "Clean Up Serialization Helpers" <<std::endl;
  network->destroy();
  engine->destroy();
  builder->destroy();
  
  // Create Inference Runtime
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr);
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  
  // Create CUDA stream for the execution of this inference.
  cudaStream_t stream;
  checkCuda( cudaStreamCreate(&stream) );
  
  std::cout << "[LOG] " << "Number of Bindings: " << engine->getNbBindings() << std::endl;
  void* buffers[2];
  
  int inputIndex = engine->getBindingIndex("InputLayer");
  int outputIndex = engine->getBindingIndex("OutputLayer");

  std::cout << "input Index: " << inputIndex << std::endl;
  std::cout << "output Index: " << outputIndex << std::endl;

  float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f};
  float h_output[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::cout << "Input: " 
            << h_input[0] << " "
            << h_input[1] << " "
            << h_input[2] << " "
            << h_input[3] << " "
            << h_input[4] << " "
            << h_input[5] << " "
            << h_input[6] << " "
            << std::endl;

  checkCuda( cudaMalloc(&buffers[inputIndex], numberInputs * sizeof(float)) );
  checkCuda( cudaMalloc(&buffers[outputIndex], numberInputs * sizeof(float)) );
  checkCuda( cudaMemcpyAsync(buffers[inputIndex], &h_input, numberInputs * sizeof(float), cudaMemcpyHostToDevice, stream) );
  if(context->enqueue(1, buffers, stream, nullptr))
    std::cout << "[LOG] " << "inference OK" << std::endl;
  else
    std::cout << "[LOG] " << "inference PROBLEM" << std::endl;
  checkCuda( cudaMemcpyAsync(&h_output, buffers[outputIndex], numberInputs * sizeof(float), cudaMemcpyDeviceToHost, stream) );
  checkCuda( cudaStreamSynchronize(stream) );

  std::cout << "Result: " 
            << h_output[0] << " "
            << h_output[1] << " "
            << h_output[2] << " "
            << h_output[3] << " "
            << h_output[4] << " "
            << h_output[5] << " "
            << h_output[6] << " "
            << std::endl;

  std::cout << "[LOG] " << "Clean Up Inference System" <<std::endl;
  context->destroy();
  engine->destroy();
  runtime->destroy();

  checkCuda( cudaStreamDestroy(stream) );
  return EXIT_SUCCESS;
}
