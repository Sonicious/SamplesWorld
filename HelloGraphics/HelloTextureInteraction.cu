#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// GL indices for vertice attributes
constexpr unsigned int GL_VERTEX_POSITION_ATTRIBUTE_IDX=0;
constexpr unsigned int GL_VERTEX_COLOR_ATTRIBUTE_IDX=1;
constexpr unsigned int GL_VERTEX_TEXTURE_ATTRIBUTE_IDX = 2;

// cuda error checking function
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// function to read files for reading shader sources
std::string readFile(const char* filePath)
{
  std::string content;
  std::ifstream fileStream(filePath, std::ios::in);

  if (!fileStream.is_open()) {
    std::cerr << "Could not read file " << filePath << std::endl << "File does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line = "";
  while (!fileStream.eof())
  {
    std::getline(fileStream, line);
    content.append(line + "\n");
  }

  fileStream.close();
  return content;
}

// function to help with Open GL Debugging
void checkGL()
{
  GLenum error = glGetError();
  if(error != GL_NO_ERROR)
  {
    printf("[ERROR] %d\n", error);
    exit(EXIT_FAILURE);
  }
  else
  {
    printf("[LOG] no error so far\n");
  }
}

// standard GLFW Error Callback
void glfwErrorCallback(int error, const char* description)
{
  printf("Error: %s\n", description);
}

// This struct is used for graphical data of triangles and according attributes
typedef struct {
  GLfloat position[3];
  GLfloat color[4];
  GLfloat tex[2];
} VertexData;

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on Mac retina displays.
    glViewport(0, 0, width, height);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA Kernels for image:

__global__ void myTextureKernel( uchar4 *texel, unsigned int width, unsigned int height, size_t pitch, double time)
{
  for (unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    idy < height;
    idy += blockDim.y * gridDim.y)
  {
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < width;
      idx += blockDim.x * gridDim.x)
    {
      int fulltime = int(time*10)%10;
      // according to CUDA documentation (see cudaMallocPitch())
      uchar4* localTexel = (uchar4*)((char*)texel + idy * pitch) + idx;
      //printf("(%u,%u) = (%u,%u,%u,%u)\n",idx, idy, localTexel->x, localTexel->y, localTexel->z, localTexel->w);
      localTexel->x = 25 * fulltime;
      localTexel->y = idx * 25 * fulltime;
      localTexel->z = idy * 25 * fulltime;
      localTexel->w = 255;
    }
  }
}

int main(int argc, char *argv[])
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
  glfwSetErrorCallback(glfwErrorCallback);
  // We want OpenGL 3.3 Core Profile
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // Create a windowed mode window and its OpenGL context
  GLFWwindow* window = glfwCreateWindow(800, 600, "Hello Cuda Texture Pitched", NULL, NULL);
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
  // Manage Callbacks for resizing:
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  // disable Vsync
  glfwSwapInterval(0);

  // get original context dimensions
  GLint contextDims[4] = {0};
  glGetIntegerv(GL_VIEWPORT, contextDims);
  GLint contextWidth = contextDims[2];
  GLint contextHeight = contextDims[3];
  printf("[LOG] Size: %d x %d\n", contextWidth, contextHeight);
  glViewport(0, 0, contextWidth, contextHeight);

///////////////////////////////////////////////////////////////////////////////
// Create Shader Program:

  std::string vertexShaderSourceString = readFile("shaders/VertexShader.glsl");
  std::string fragmentShaderColorSourceString = readFile("shaders/FragmentColorShader.glsl");
  std::string fragmentShaderTextureSourceString = readFile("shaders/FragmentTextureShader.glsl");
  const GLchar* vertexShaderSource = vertexShaderSourceString.c_str();
  const GLchar* fragmentShaderColorSource = fragmentShaderColorSourceString.c_str();
  const GLchar* fragmentShaderTextureSource = fragmentShaderTextureSourceString.c_str();
  
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
  // Create color fragment shader, load and compile
  GLuint fragmentColorShader;
  fragmentColorShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentColorShader, 1, &fragmentShaderColorSource, NULL);
  glCompileShader(fragmentColorShader);
  // check for errors in compilation process
  glGetShaderiv(fragmentColorShader, GL_COMPILE_STATUS, &success);
  if(!success)
  {
    glGetShaderInfoLog(fragmentColorShader, 512, NULL, infoLog);
    printf("Fragment Color Shader Compilation Failed:\n%s\n",infoLog);
    return EXIT_FAILURE;
  }
  // Create texture fragment shader, load and compile
  GLuint fragmentTextureShader;
  fragmentTextureShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentTextureShader, 1, &fragmentShaderTextureSource, NULL);
  glCompileShader(fragmentTextureShader);
  // check for errors in compilation process
  glGetShaderiv(fragmentTextureShader, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    glGetShaderInfoLog(fragmentTextureShader, 512, NULL, infoLog);
    printf("Fragment Texture Shader Compilation Failed:\n%s\n", infoLog);
    return EXIT_FAILURE;
  }
  // Link Shaders together to a program
  GLuint triangleShaderProgram;
  triangleShaderProgram = glCreateProgram();
  glAttachShader(triangleShaderProgram, vertexShader);
  glAttachShader(triangleShaderProgram, fragmentColorShader);
  glLinkProgram(triangleShaderProgram);
  glGetProgramiv(triangleShaderProgram, GL_LINK_STATUS, &success);
  if(!success)
  {
      glGetProgramInfoLog(triangleShaderProgram, 512, NULL, infoLog);
      printf("Program Linking Failed:\n%s\n",infoLog);
      return EXIT_FAILURE;
  }
  // Same for the textureShader for the ground
  GLuint groundShaderProgram;
  groundShaderProgram = glCreateProgram();
  glAttachShader(groundShaderProgram, vertexShader);
  glAttachShader(groundShaderProgram, fragmentTextureShader);
  glLinkProgram(groundShaderProgram);
  glGetProgramiv(groundShaderProgram, GL_LINK_STATUS, &success);
  if (!success)
  {
    glGetProgramInfoLog(groundShaderProgram, 512, NULL, infoLog);
    printf("Problem with linking appeared:\n%s\n", infoLog);
    return EXIT_FAILURE;
  }

  // delete Shaders (not needed anymore, because completely compiled and so on)
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentColorShader);
  glDeleteShader(fragmentTextureShader);

///////////////////////////////////////////////////////////////////////////////
// Create a state driven VAO Setup Vertex Data, Buffers and
// configure Vertex Attributes

  // Create an bind a VAO
  GLuint triangleVAO; // vertex array object
  glGenVertexArrays(1, &triangleVAO);
  glBindVertexArray(triangleVAO); // Bind Vertex Array

  // generate buffer and Array for vertices and bind and fill it
  GLuint trianglePositionsVBO; // Vertex Buffer Object
  glGenBuffers(1, &trianglePositionsVBO);
  glBindBuffer(GL_ARRAY_BUFFER, trianglePositionsVBO);
  VertexData myTriangle[3] =
  {
    {{-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{ 1.0f, 0.0f,-1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{ 1.0f, 1.5f,-1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}
  };
  // Upload vertex data to GPU
  glBufferData(GL_ARRAY_BUFFER, sizeof(myTriangle), myTriangle, GL_STATIC_DRAW);
  // Explain Data via VertexAttributePointers to the shader
  // First enable stream index, then submit information
  glEnableVertexAttribArray(GL_VERTEX_POSITION_ATTRIBUTE_IDX);
  glVertexAttribPointer(GL_VERTEX_POSITION_ATTRIBUTE_IDX, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*) offsetof(VertexData,position));
  // Same for Color: Pay Attention of Stride and begin
  glEnableVertexAttribArray(GL_VERTEX_COLOR_ATTRIBUTE_IDX);
  glVertexAttribPointer(GL_VERTEX_COLOR_ATTRIBUTE_IDX, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*) offsetof(VertexData, color)); // offsetof(VertexData, color) benutzen wegen struct padding
  // possible unbinding:
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // The ground should be textured
  GLuint groundTexture;
  glActiveTexture(GL_TEXTURE0);
  glGenTextures(1, &groundTexture);
  glBindTexture(GL_TEXTURE_2D, groundTexture);
  GLubyte textureData[] = // just 4 different colors
  {
    255, 255, 255, 255, 0, 0, 0, 255,
    255, 0, 0, 255, 0, 0, 255, 255
  };
  GLuint textureWidth = 2, textureHeight = 2;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
  glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);

  GLuint groundVAO, groundPositionsVBO;
  glGenVertexArrays(1, &groundVAO); glGenBuffers(1, &groundPositionsVBO);
  VertexData myGround[6] =
  {
    {{ -1.0f, 0.0f,  1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{ -1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{  1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{ -1.0f, 0.0f,  1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{  1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{  1.0f, 0.0f,  1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}}
  };
  glBindVertexArray(groundVAO);
  glBindBuffer(GL_ARRAY_BUFFER, groundPositionsVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(myGround), myGround, GL_STATIC_DRAW);
  glEnableVertexAttribArray(GL_VERTEX_POSITION_ATTRIBUTE_IDX);
  glVertexAttribPointer(GL_VERTEX_POSITION_ATTRIBUTE_IDX, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*)offsetof(VertexData, position));
  glEnableVertexAttribArray(GL_VERTEX_COLOR_ATTRIBUTE_IDX);
  glVertexAttribPointer(GL_VERTEX_COLOR_ATTRIBUTE_IDX, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*)offsetof(VertexData, color));
  glEnableVertexAttribArray(GL_VERTEX_TEXTURE_ATTRIBUTE_IDX);
  glVertexAttribPointer(GL_VERTEX_TEXTURE_ATTRIBUTE_IDX, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*)offsetof(VertexData, tex));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  

  ///////////////////////////////////////////////////////////////////////////////
  // Projection

  // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
  glm::mat4 projectionMat = glm::perspective(glm::radians(45.0f), (float)4.0 / (float)3.0, 0.1f, 100.0f);
  // Or, for an ortho camera :
  //glm::mat4 Projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates

  // Camera matrix
  glm::mat4 viewMat = glm::lookAt(
    glm::vec3(1, 2, 4), // Camera is at (1,1,4), in World Space
    glm::vec3(0, 0, 0), // and looks at this point
    glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
  );

  // Model matrix : an identity matrix (model will be at the origin)
  glm::mat4 triangleModelMat = glm::mat4(1.0f);
  glm::mat4 groundModelMat = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
  // Our ModelViewProjection : multiplication of our 3 matrices
  glm::mat4 triangleMvpMat = projectionMat * viewMat * triangleModelMat; // Remember, matrix multiplication is the other way around
  glm::mat4 groundMvpMat = projectionMat * viewMat * groundModelMat;
  // Get important Handles from OpenGL
  GLuint triangleMatrixId = glGetUniformLocation(triangleShaderProgram, "MVP");
  GLuint groundMatrixId = glGetUniformLocation(groundShaderProgram, "MVP");
  GLuint textureId = glGetUniformLocation(groundShaderProgram, "mytexture");
  
  ///////////////////////////////////////////////////////////////////////////////
  // Last Setup points
   
  // Dark blue background
  glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
  // Enable depth test
  glEnable(GL_DEPTH_TEST);
  // Accept fragment if it closer to the camera than the former one
  glDepthFunc(GL_LESS);
  // uncomment this call to draw in wireframe polygons.
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  ///////////////////////////////////////////////////////////////////////////////
  // Cuda Texture Interop

  // Register the TextureImage to be accessed by CUDA
  // no special flags given
  cudaGraphicsResource_t groundTextureGraphicResource;
  cudaChannelFormatDesc groundTextureFormatDesc;
  cudaExtent groundTextureExtend;
  unsigned int groundTextureFlags;
  cudaArray_t groundTextureCudaArray;
  checkCuda(cudaGraphicsGLRegisterImage(
    &groundTextureGraphicResource,
    groundTexture,
    GL_TEXTURE_2D,
    cudaGraphicsRegisterFlagsNone
  ));
  uchar4 *d_groundTexturePitchedMemory;
  size_t groundTexturePitch;
  // create a 2D memory with pitch for holding the texture
  checkCuda(cudaMallocPitch(&d_groundTexturePitchedMemory, &groundTexturePitch, 2 * sizeof(uchar4), 2));

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
        sprintf(windowTitle, "Hello Cuda Texture Pitched | %f ms/frame", 1000.0/double(nbFrames));
        glfwSetWindowTitle(window, windowTitle);
        nbFrames = 0;
        lastTime += 1.0;
    }

    // map ressource. Now don't interact with it via OpenGL
    checkCuda(cudaGraphicsMapResources(1, &groundTextureGraphicResource, 0));
    checkCuda(cudaGraphicsSubResourceGetMappedArray(&groundTextureCudaArray, groundTextureGraphicResource, 0, 0));
    //checkCuda(cudaArrayGetInfo(&groundTextureFormatDesc, &groundTextureExtend, &groundTextureFlags, groundTextureCudaArray));
    // Here some possible information about the texture. groundTextureFormatDesc.f: cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat, cudaChannelFormatKindNone
    //std::cout << "TextureShape: " << groundTextureExtend.width << "x" << groundTextureExtend.height << "x" << groundTextureExtend.depth <<
    //  " FormatType: " << groundTextureFormatDesc.x << "x" << groundTextureFormatDesc.y << "x" << groundTextureFormatDesc.z << "x" << groundTextureFormatDesc.w << "x" << groundTextureFormatDesc.f << std::endl;    
    // copy from Array to 2D memory (exact size must be known here, but can be derived through cudaArrayGetInfo )
    checkCuda(cudaMemcpy2DFromArray(d_groundTexturePitchedMemory, groundTexturePitch, groundTextureCudaArray, 0, 0, 2*sizeof(uchar4), 2, cudaMemcpyDeviceToDevice));
    // cuda Kernel here to deal with everything
    myTextureKernel <<<dim3(1, 1), dim3(2,2)>>>(d_groundTexturePitchedMemory, 2, 2, groundTexturePitch, glfwGetTime());
    // copy 2D memory back to texture
    checkCuda(cudaMemcpy2DToArray(groundTextureCudaArray, 0, 0, d_groundTexturePitchedMemory, groundTexturePitch, 2 * sizeof(uchar4), 2, cudaMemcpyDeviceToDevice));
    // unmap ressource. Now you can do stuff with it again
    checkCuda(cudaGraphicsUnmapResources(1, &groundTextureGraphicResource, 0));

    // Clear the screen and the z Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw the triangle
    glBindVertexArray(triangleVAO); // Bind the vertex array object and all its state
    glUseProgram(triangleShaderProgram); // activate the shaders
    glUniformMatrix4fv(triangleMatrixId, 1, GL_FALSE, glm::value_ptr(triangleMvpMat)); // update the MVP in the currently activated shader
    glDrawArrays(GL_TRIANGLES, 0, 3); // draw the vertices as single triangle according to program
    
    // draw Ground
    glBindVertexArray(groundVAO);
    glActiveTexture(GL_TEXTURE0); // activate texture unit
    glBindTexture(GL_TEXTURE_2D, groundTexture); // bind texture to texture unit
    glUseProgram(groundShaderProgram);
    glUniformMatrix4fv(groundMatrixId, 1, GL_FALSE, glm::value_ptr(groundMvpMat));
    glDrawArrays(GL_TRIANGLES, 0,6);

    // unbind everything
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    
    // Swap front and back buffers
    glfwSwapBuffers(window);
    // Check for Inputs:
    processInput(window);
    // Poll for and process events
    glfwPollEvents();
  }

  // Cleanup
  checkCuda(cudaFree(d_groundTexturePitchedMemory));
  // OpenGL is reference counted and terminated by GLFW
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
