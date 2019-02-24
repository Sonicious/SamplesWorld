#include <cstdlib>
#include <cstdio>
#include <iostream>

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <GL/gl.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

inline void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("[ERROR] %s\n", cudaGetErrorName(result));
        printf("[ERROR] %s\n", cudaGetErrorString(result));
    }
}

///////////////////////////////////////////////////////////////////////////////
// CUDA Kernel for uniform image:

__global__ void myKernel(float *renderImageData, size_t size)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x) 
      {
          renderImageData[i] = 1.0f;
      }
}

///////////////////////////////////////////////////////////////////////////////
// IO-Callbacks:

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// Vertex Shader Source:
// input comes to "in vec3 aPos"
// output goes to "gl_position
const GLchar *vertexShaderSource = "#version 330 core\n"
  "layout (location = 0) in vec3 aPos;\n"
  "layout (location = 1) in vec2 aTexCoord;\n"
  "out vec2 TexCoord;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
  "   TexCoord = aTexCoord;\n"
  "}\0";

// Fragment Shader Source:
// out declares output
// output is always FragColor
const GLchar *fragmentShaderSource = "#version 330 core\n"
  "out vec4 FragColor;\n"
  "in vec2 TexCoord;\n"
  "uniform sampler2D ourTexture;\n"
  "void main()\n"
  "{\n"
  "   FragColor = texture(ourTexture, TexCoord);\n"
  "}\n\0";

void error_callback(int error, const char* description)
{
  printf("Error: %s\n", description);
}

int main(void)
{
  // OpenGL Status Variables:
  GLint  success;
  char infoLog[512];

///////////////////////////////////////////////////////////////////////////////
// Initialize everything for GLFW

  // Initialize the library
  if (!glfwInit())
  {
    return EXIT_FAILURE;
  }
  // Set Error Callback
  glfwSetErrorCallback(error_callback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // Create a windowed mode window and its OpenGL context
  GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
  if (!window)
  {
    printf("Failed to create GLFW window!");
    glfwTerminate();
    return EXIT_FAILURE;
  }
  // Make the window's context current
  glfwMakeContextCurrent(window);
  // Manage Callbacks:
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

///////////////////////////////////////////////////////////////////////////////
// Create Shader Program:
  
  // Create vertex shader and load and compile source
  GLuint vertexShader;
  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  // check for errors in compilation process
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if(!success)
  {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    //printf("Vertex Shader Compilation Failed:\n%s\n",infoLog);
  }
  // Create fragment shader, load and compile
  GLuint fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  // check for errors in compilation process
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if(!success)
  {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    printf("Fragment Shader Compilation Failed:\n%s\n",infoLog);
  }
  // Link Shaders together to a program
  GLuint shaderProgram;
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if(!success)
  {
      glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
      printf("Program Linking Failed:\n%s\n",infoLog);
  }
  // Use the program for the pipeline
  glUseProgram(shaderProgram);
  // delete Shaders (not needed anymore)
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

///////////////////////////////////////////////////////////////////////////////
// Setup Vertex Data, Buffers and configure Vertex Attributes:

  float vertices[] = {
   // positions        // text-coords
   -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // SW
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // SE
   -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, // NW
    1.0f,  1.0f, 0.0f, 1.0f, 1.0f  // NE
  };

  // generate buffer and Array for vertices and bind and fill it
  GLuint VBO, VAO;// Vertex Buffer Object, Vertex Array Object
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO); // Bind Vertex Array First
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // Copy Vertices Data
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // Explain Data via VertexAttributePointers
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0);
  // By Default, it's disabled
  // Enable the Vertex Attribute
  glEnableVertexAttribArray(0);
  // Same for Texture: Pay Attention of Stride and begin
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // possible unbinding:
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

///////////////////////////////////////////////////////////////////////////////
// Texture stuff

  // Black/white checkerboard
  float pixels[] = {
      1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f
  };

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  // set the texture wrapping/filtering options (on the currently bound texture object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);
  glBindTexture(GL_TEXTURE_2D, 0);

///////////////////////////////////////////////////////////////////////////////
// CUDA Texture stuff!

  printf("Real Interop Starts\n");
  GLuint interopTexture;
  float *deviceRenderBuffer;
  float *hostRenderBuffer;
  cudaGraphicsResource *textureGraphicResource;
  cudaArray_t textureCudaArray;

  //MemAlloc Cuda Buffer
  int numTexels = 2*2;
  int numValues = numTexels*4; // RGBA
  size_t sizeTexData = numValues * sizeof(GLfloat);
  
  printf("HostMalloc\n");
  hostRenderBuffer = (float*)malloc(sizeTexData);
  printf("CudaMalloc\n");
  checkCuda( cudaMalloc(&deviceRenderBuffer, sizeTexData) );
  // Here the Calculations for the interop-Data
  printf("Kernel starts\n");
  myKernel<<<1,1>>>(deviceRenderBuffer, sizeTexData);
  cudaMemcpy(hostRenderBuffer, deviceRenderBuffer, sizeTexData, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < numValues; i++ )
  {
    printf("%f\n",hostRenderBuffer[i]);
  }

  glGenTextures(1, &interopTexture);
  glBindTexture(GL_TEXTURE_2D, interopTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // Just allocate, but no copy to it:
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, NULL);

  // Register Resource to texture
  printf("GLRegister\n");
  checkCuda( cudaGraphicsGLRegisterImage( &textureGraphicResource, interopTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

  // The next lines should go to Render-Loop:
  // Map: now just changing inside Default Stream 0
  printf("Mapping\n");
  cudaDeviceSynchronize();
  checkCuda( cudaGraphicsMapResources(1, &textureGraphicResource) );
  // get the corresponding CudaArray of Resource at array position 0 and mipmap level 0
  printf("get Mapped Array\n");
  checkCuda( cudaGraphicsSubResourceGetMappedArray(&textureCudaArray, textureGraphicResource, 0, 0) );
  // copy Data to CudaArray from deviceRenderBuffer, wOffset=0, hOffset=0
  
  // Test:
  checkCuda( cudaMemcpyFromArray(hostRenderBuffer, textureCudaArray, 0, 0, sizeTexData, cudaMemcpyDeviceToHost) );
  for (size_t i = 0; i < numValues; i++ )
  {
    printf("%f\n",hostRenderBuffer[i]);
  }

  printf("MemCopy to Array\n");
  checkCuda( cudaMemcpyToArray(textureCudaArray, 0, 0, (void*)deviceRenderBuffer, sizeTexData, cudaMemcpyDeviceToDevice) );
  // Unmap the resource from Stream 0
  cudaDeviceSynchronize();
  printf("Unmapping\n");
  checkCuda( cudaGraphicsUnmapResources(1, &textureGraphicResource) );

///////////////////////////////////////////////////////////////////////////////
// Render Loop
  while (!glfwWindowShouldClose(window))
  {
    // set bg color here via Clearing
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set Up Texture
    glBindTexture(GL_TEXTURE_2D, interopTexture);

    // Draw 
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // All about the loaded VAO
    // glBindVertexArray(0); // To unbind Vertex Array

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // CHeck for Inputs:
    processInput(window);
    // Poll for and process events
    glfwPollEvents();
  }

  // Some more Cleanup Probably of CUDA stuff
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
