#include <cstdlib>
#include <cstdio>
#include <iostream>

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <GL/gl.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// GL Defines for stuff
#define GL_VERTEX_POSITION_ATTRIBUTE 0
#define GL_VERTEX_TEXTURE_COORD_ATTRIBUTE 1

inline void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("[ERROR] %s\n", cudaGetErrorName(result));
        printf("[ERROR] %s\n", cudaGetErrorString(result));
    }
}

///////////////////////////////////////////////////////////////////////////////
// CUDA Kernel for image:

__global__ void myKernel(unsigned char *renderImageData, size_t size)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x) 
      {
          renderImageData[i] = (i/4) % 255;
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
  GLFWwindow* window = glfwCreateWindow(640, 480, "Hello Cuda GLFW Interop", NULL, NULL);
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

  // delete Shaders (not needed anymore, because completely compiled and so on)
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

///////////////////////////////////////////////////////////////////////////////
// Setup Vertex Data, Buffers and configure Vertex Attributes:

  float vertices[] = {
   // pos (x,y,z)      // text-coords (u,v)
   -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // SW
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // SE
   -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, // NW
    1.0f,  1.0f, 0.0f, 1.0f, 1.0f  // NE
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
  unsigned char *deviceRenderBuffer;
  cudaGraphicsResource *textureGraphicResource;
  cudaArray *textureCudaArray;


  // calculate Data size and MemAlloc Cuda Buffer
  int textureWidth = 16;
  int textureHeight = 16;
  int numTexels = textureHeight * textureWidth;
  int numValues = numTexels*4; // RGBA
  size_t sizeTexData = numValues * sizeof(GLubyte);
  checkCuda( cudaMalloc(&deviceRenderBuffer, sizeTexData) );
  
  // Here the Calculations for the interop-Data
  glGenTextures(1, &interopTexture);
  glBindTexture(GL_TEXTURE_2D, interopTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // Just allocate, but no copy to it:
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  // Register Resource to texture
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
    myKernel<<<2, 2>>>(deviceRenderBuffer, sizeTexData);
    // Map graphics resource for access by CUDA
    checkCuda( cudaGraphicsMapResources(1, &textureGraphicResource, 0) );
    // get the corresponding CudaArray of Resource at array position 0 and mipmap level 0
    checkCuda( cudaGraphicsSubResourceGetMappedArray(&textureCudaArray, textureGraphicResource, 0, 0) );
    // copy Data to CudaArray from deviceRenderBuffer, wOffset=0, hOffset=0
    //checkCuda( cudaMemcpyToArray(textureCudaArray, 0, 0, deviceRenderBuffer, sizeTexData, cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy2DToArray(textureCudaArray, 0, 0, deviceRenderBuffer, textureWidth*4, textureWidth*4, textureHeight, cudaMemcpyDeviceToDevice));
    // Unmap the resource from Stream 0
    checkCuda( cudaGraphicsUnmapResources(1, &textureGraphicResource, 0) );

    // Draw
    // Use the program for the pipeline (keep it to save state to VAO)
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO); // Program is bound to VAO
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
  checkCuda( cudaFree(deviceRenderBuffer) );
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
