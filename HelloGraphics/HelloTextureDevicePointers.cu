// other Headers
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// GL Defines for stuff
constexpr int GL_VERTEX_POSITION_ATTRIBUTE = 0;
constexpr int GL_VERTEX_TEXTURE_COORD_ATTRIBUTE = 1;

void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("[ERROR] %s\n", cudaGetErrorName(result));
        printf("[ERROR] %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void checkGL()
{
  GLenum error = glGetError();
  if(error != GL_NO_ERROR)
  {
    printf("[ERROR] %d\n", error);
    exit(EXIT_FAILURE);
  }
}

void glfwErrorCallback(int error, const char* description)
{
  printf("Error: %s\n", description);
}

typedef struct {
  GLfloat position[3];
  GLfloat textureCoords[2];
} VertexData;

typedef struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
} pixelRGBA;

///////////////////////////////////////////////////////////////////////////////
// CUDA Kernel for image:

__global__ void myTextureKernel(pixelRGBA *renderImageData, size_t width, size_t height, size_t pitch)
{
  for (int idy = blockIdx.y * blockDim.y + threadIdx.y;
         idy < height;
         idy += blockDim.y * gridDim.y)
      {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
               idx < width;
               idx += blockDim.x * gridDim.x) 
            {
                // according to CUDA documentation (see cudaMallocPitch())
                pixelRGBA *myPixel = (pixelRGBA*) ((char*)renderImageData + idy*pitch) + idx;
                myPixel->r = 128;
                myPixel->g = 120;
                myPixel->b = 80;
                myPixel->a = 255;
            }
      }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void FramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// Vertex Shader Source:
// input comes to "in vec3 aPos"
// output goes to "gl_position
const GLchar *vertexShaderSource =
  "#version 450 core\n"
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
const GLchar *fragmentShaderSource =
  "#version 450 core\n"
  "out vec4 FragColor;\n"
  "in vec2 TexCoord;\n"
  "uniform sampler2D ourTexture;\n"
  "void main()\n"
  "{\n"
  "   FragColor = texture(ourTexture, TexCoord);\n"
  "}\n\0";

int main(int argc, char *argv[])
{
  // read TextureSize
  int textureWidth = 128;
  int textureHeight = 128;
  if (argc >= 3)
  {
    textureWidth = atoi(argv[1]);
    textureHeight = atoi(argv[2]);
  }

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
  glfwSetErrorCallback(glfwErrorCallback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 4.5
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // Create a windowed mode window and its OpenGL context
  GLFWwindow* window = glfwCreateWindow(1000, 1000, "Hello Cuda GLFW Interop", NULL, NULL);
  if (!window)
  {
    printf("Failed to create GLFW window!");
    glfwTerminate();
    return EXIT_FAILURE;
  }
  // Make the window's context current
  glfwMakeContextCurrent(window);
  // load pointers to OpenGL functions at runtime
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
  {
      printf("Failed to initialize OpenGL context");
      return EXIT_FAILURE;
  }
  // Manage Callbacks:
  glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);
    // disable Vsync
  glfwSwapInterval(0);

///////////////////////////////////////////////////////////////////////////////
// Create a state driven VAO
  GLuint VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO); // Bind Vertex Array First

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
    printf("Vertex Shader Compilation Failed:\n%s\n",infoLog);
    return EXIT_FAILURE;
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
    return EXIT_FAILURE;
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
      return EXIT_FAILURE;
  }

  // delete Shaders (not needed anymore, because completely compiled and so on)
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

///////////////////////////////////////////////////////////////////////////////
// Setup Vertex Data, Buffers and configure Vertex Attributes:

  VertexData vertices[] = {
    {{-1.0f, -1.0f, 0.0f},{0.0f, 0.0f}},
    {{ 1.0f, -1.0f, 0.0f},{1.0f, 0.0f}},
    {{-1.0f,  1.0f, 0.0f},{0.0f, 1.0f}},
    {{ 1.0f,  1.0f, 0.0f},{1.0f, 1.0f}}
  };
  // generate buffer and Array for vertices and bind and fill it
  GLuint VBO;// Vertex Buffer Object, Vertex Array Object
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // Copy Vertices Data
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // Explain Data via VertexAttributePointers to the shader
  // By Default, it's disabled
  // Enable the Vertex Attribute
  glEnableVertexAttribArray(GL_VERTEX_POSITION_ATTRIBUTE);
  glVertexAttribPointer(GL_VERTEX_POSITION_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0);
  // Same for Texture: Pay Attention of Stride and begin
  glEnableVertexAttribArray(GL_VERTEX_TEXTURE_COORD_ATTRIBUTE);
  glVertexAttribPointer(GL_VERTEX_TEXTURE_COORD_ATTRIBUTE, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

  // possible unbinding:
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

///////////////////////////////////////////////////////////////////////////////
// CUDA Texture Interaction

  GLuint interopTexture;
  pixelRGBA *deviceTextureGraphic;
  size_t deviceTextureGraphicPitch;
  cudaGraphicsResource_t textureGraphicResource = 0;
  cudaArray *textureCudaArray;


  // calculate Data size and MemAlloc Cuda Buffer
  checkCuda( cudaMallocPitch(&deviceTextureGraphic, &deviceTextureGraphicPitch, textureWidth * sizeof(pixelRGBA), textureHeight) );
  
  // Here the Calculations for the interop-Data
  glGenTextures(1, &interopTexture);
  glBindTexture(GL_TEXTURE_2D, interopTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // Just allocate, but no copy to it:
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  // Register OpenGL texture to a cudaGraphicsResource for CUDA
  checkCuda( cudaGraphicsGLRegisterImage( &textureGraphicResource, interopTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

///////////////////////////////////////////////////////////////////////////////
// Render Loop
  double lastTime = glfwGetTime();
  int nbFrames = 0;
  char windowTitle[100];

  while (!glfwWindowShouldClose(window))
  {
    // Measure speed
    double currentTime = glfwGetTime();
    nbFrames++;
    if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
        // printf and reset timer
        sprintf(windowTitle, "Hello Cuda GLFW Interop | %f ms/frame", 1000.0/double(nbFrames));
        glfwSetWindowTitle(window, windowTitle);
        nbFrames = 0;
        lastTime += 1.0;
    }

    // set bg color here via Clearing
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // run CUDA
    myTextureKernel<<<16, 32>>>(deviceTextureGraphic, textureWidth, textureHeight, deviceTextureGraphicPitch);
    // Map 1 graphics resource for access by CUDA in stream 0
    checkCuda( cudaGraphicsMapResources(1, &textureGraphicResource, 0) );
    // get the corresponding CudaArray of Resource at array position 0 and mipmap level 0
    checkCuda( cudaGraphicsSubResourceGetMappedArray(&textureCudaArray, textureGraphicResource, 0, 0) );
    // copy Data to CudaArray from deviceRenderBuffer, wOffset=0, hOffset=0
    checkCuda( cudaMemcpy2DToArray(textureCudaArray, 0, 0, deviceTextureGraphic, deviceTextureGraphicPitch, textureWidth * sizeof(pixelRGBA), textureHeight, cudaMemcpyDeviceToDevice));
    // Unmap 1 resource from Stream 0
    checkCuda( cudaGraphicsUnmapResources(1, &textureGraphicResource, 0) );

    // Draw
    // Use the program for the pipeline (keep it to save state to VAO)
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO); checkGL(); // Program is bound to VAO
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // All about the loaded VAO
    glBindVertexArray(0); // To unbind Vertex Array
    glUseProgram(0);

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // Check for Inputs:
    processInput(window);
    // Poll for and process events
    glfwPollEvents();
  }

  // Cleanup
  // OpenGL is reference counted and terminated by GLFW
  checkCuda( cudaGraphicsUnregisterResource(textureGraphicResource) );
  checkCuda( cudaFree(deviceTextureGraphic) );
  checkCuda( cudaDeviceReset() );
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
