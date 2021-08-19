// other Headers
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// GL indices for vertice attributes
constexpr unsigned int GL_VERTEX_POSITION_ATTRIBUTE_IDX=0;
constexpr unsigned int GL_VERTEX_COLOR_ATTRIBUTE_IDX=1;

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
const GLchar *vertexShaderSource =
  "#version 450 core\n"
  "layout (location = 0) in vec3 vPos;\n" // must be idx of GL_VERTEX_POSITION_ATTRIBUTE_IDX
  "layout (location = 1) in vec4 vColor;\n" // must be idx of GL_VERTEX_COLOR_ATTRIBUTE_IDX
  "out vec4 fColor;\n"
  "uniform mat4 MVP;\n" // constant which must be uploaded from CPU
  "void main()\n"
  "{\n"
  "   gl_Position = MVP * vec4(vPos.x, vPos.y, vPos.z, 1.0);\n"
  "   fColor = vColor;\n"
  "}\0";

// Fragment Shader Source:
const GLchar *fragmentShaderSource =
  "#version 450 core\n"
  "in vec4 fColor;\n"
  "out vec4 outColor;\n"
  "void main()\n"
  "{\n"
  "   outColor = fColor;\n"
  "}\n\0";

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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // Create a windowed mode window and its OpenGL context
  GLFWwindow* window = glfwCreateWindow(800, 600, "Hello Graphics", NULL, NULL);
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
// Create a state driven VAO Setup Vertex Data, Buffers and
// configure Vertex Attributes

  // Create an bind a VAO
  GLuint vertexArrayObject; // vertex array object
  glGenVertexArrays(1, &vertexArrayObject);
  glBindVertexArray(vertexArrayObject); // Bind Vertex Array

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

  ///////////////////////////////////////////////////////////////////////////////
  // Projection

  // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
  glm::mat4 projectionMat = glm::perspective(glm::radians(45.0f), (float)4.0 / (float)3.0, 0.1f, 100.0f);
  // Or, for an ortho camera :
  //glm::mat4 Projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates

  // Camera matrix
  glm::mat4 viewMat = glm::lookAt(
    glm::vec3(1, 1, 2), // Camera is at (1,3,2), in World Space
    glm::vec3(0, 0, 0), // and looks at the origin
    glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
  );

  // Model matrix : an identity matrix (model will be at the origin)
  glm::mat4 modelMat = glm::mat4(1.0f);
  // Our ModelViewProjection : multiplication of our 3 matrices
  glm::mat4 mvpMat = projectionMat * viewMat * modelMat; // Remember, matrix multiplication is the other way around
  // Get MVP Handle from Program
  GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");
  
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

    // Clear the screen and the z Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Bind the vertex array object and all its state
    glBindVertexArray(vertexArrayObject);
    // activate the shaders
    glUseProgram(shaderProgram);
    // update the MVP in the currently activated shader
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvpMat[0][0]);
    // draw the vertices as single triangles according to program
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // unbind everything
    glBindVertexArray(0);
    glUseProgram(0);
    
    // Swap front and back buffers
    glfwSwapBuffers(window);
    // Check for Inputs:
    processInput(window);
    // Poll for and process events
    glfwPollEvents();
  }

  checkGL();
  // Cleanup
  // OpenGL is reference counted and terminated by GLFW
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
