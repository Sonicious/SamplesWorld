#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

// GL Defines for stuff
#define GL_VERTEX_POSITION_ATTRIBUTE 0

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

// glfw: Error Callback for GLFW
// -----------------------------
void error_callback(int error, const char* description)
{
  printf("Error: %s\n", description);
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

// Vertex Shader Source:
// input comes to "in vec3 aPos"
// output goes to "gl_position
const GLchar *vertexShaderSource = "#version 330 core\n"
  "layout (location = 0) in vec3 aPos;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
  "}\0";

// Fragment Shader Source:
// out declares output
// output is always FragColor
const GLchar *fragmentShaderSource = "#version 330 core\n"
  "out vec4 FragColor;\n"
  "void main()\n"
  "{\n"
  "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
  "}\n\0";

int main(void)
{
  // OpenGL Status Variables:
  GLint  success;
  char infoLog[512];

///////////////////////////////////////////////////////////////////////////////
// Initialize everything till Rendering-Loop:

  // Initialize the library
  if (!glfwInit())
  {
    return EXIT_FAILURE;
  }
  // Set Error Callback
  glfwSetErrorCallback(error_callback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.2
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
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
  // load pointers to OpenGL functions at runtime
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
  {
      printf("Failed to initialize OpenGL context");
      return EXIT_FAILURE;
  }
  // Manage Callbacks:
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

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
   -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f,  1.0f, 0.0f
  };

  // generate buffer and Array for vertices and bind and fill it
  GLuint VBO, VAO;// Vertex Buffer Object, Vertex Array Object
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO); // Bind Vertex Array First
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // Copy Vertices Data
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // Enable the Vertex Attribute
  glEnableVertexAttribArray(GL_VERTEX_POSITION_ATTRIBUTE); 
  // Explain Data via VertexAttributePointers
  glVertexAttribPointer(GL_VERTEX_POSITION_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (GLvoid*)0);

  // possible unbinding:
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

///////////////////////////////////////////////////////////////////////////////
// Render Loop
  while (!glfwWindowShouldClose(window))
  {
    // set bg color here via Clearing
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw 
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3); // All about the loaded VAO
    // glBindVertexArray(0); // To unbind Vertex Array

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // CHeck for Inputs:
    processInput(window);
    // Poll for and process events
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
