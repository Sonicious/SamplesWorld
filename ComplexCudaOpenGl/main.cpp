#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <GL/gl.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// GL Defines for stuff
#define GL_VERTEX_POSITION_ATTRIBUTE 0
#define GL_VERTEX_COLOR_ATTRIBUTE 1

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
  GLfloat color[4];
} VertexData;

///////////////////////////////////////////////////////////////////////////////
// CUDA Kernel for image:

__global__ void myPixelBufferKernel(double time)
{
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
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// Vertex Shader Source:
// input comes to "in vec3 aPos"
// output goes to "gl_position
const GLchar *vertexShaderSource = "#version 330 core\n"
  "layout (location = 0) in vec3 vPos;\n"
  "layout (location = 1) in vec4 vColor;\n"
  "out vec4 fColor;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = vec4(vPos.x, vPos.y, vPos.z, 1.0);\n"
  "   fColor = vColor;\n"
  "}\0";

// Fragment Shader Source:
// out declares output
// output is always FragColor
const GLchar *fragmentShaderSource = "#version 330 core\n"
  "out vec4 outColor;\n"
  "in vec4 fColor;\n"
  "void main()\n"
  "{\n"
  "   outColor = fColor;\n"
  "}\n\0";

int main(int argc, char *argv[])
{
  // read Speed
  if (argc >= 1)
  {
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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  // disable Vsync
  glfwSwapInterval(0);

///////////////////////////////////////////////////////////////////////////////
// Create a state driven VAO
  GLuint VAO; // vertex array object
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
// Setup Vertex Data, Buffers and configure Vertex Attributes

  // generate buffer and Array for vertices and bind and fill it
  GLuint glPositionsVBO; // Vertex Buffer Object
  glGenBuffers(1, &glPositionsVBO);
  glBindBuffer(GL_ARRAY_BUFFER, glPositionsVBO);
  VertexData myTriangle[3] = 
  {
    {{-0.8f, -0.8f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{ 0.8f, -0.8f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    {{ 0.0f,  0.8f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}
  };
  // Allocate Vertices Data
  glBufferData(GL_ARRAY_BUFFER, sizeof(myTriangle), myTriangle, GL_STATIC_DRAW);
  // Explain Data via VertexAttributePointers to the shader
  // By Default, it's disabled
  // Enable the Vertex Attribute
  glEnableVertexAttribArray(GL_VERTEX_POSITION_ATTRIBUTE);
  glVertexAttribPointer(GL_VERTEX_POSITION_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*)0);
  // Same for Texture: Pay Attention of Stride and begin
  glEnableVertexAttribArray(GL_VERTEX_COLOR_ATTRIBUTE);
  glVertexAttribPointer(GL_VERTEX_COLOR_ATTRIBUTE, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (GLvoid*)(3 * sizeof(float)));

  // possible unbinding:
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
///////////////////////////////////////////////////////////////////////////////
// Cuda Pixel Buffer Interop


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


    // Draw
    // Use the program for the pipeline (keep it to save state to VAO)
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO); // Program is bound to VAO
    glDrawArrays(GL_TRIANGLES, 0, 3); // All about the loaded VAO
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
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
