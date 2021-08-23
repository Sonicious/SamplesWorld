// other Headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// GL indices for vertice attributes
constexpr unsigned int GL_VERTEX_POSITION_ATTRIBUTE_IDX=0;
constexpr unsigned int GL_VERTEX_COLOR_ATTRIBUTE_IDX=1;
constexpr unsigned int GL_VERTEX_TEXTURE_ATTRIBUTE_IDX = 2;

std::string readFile(const char* filePath) {
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

void glfwErrorCallback(int error, const char* description)
{
  printf("Error: %s\n", description);
}

typedef struct {
  GLfloat position[3];
  GLfloat color[4];
  GLfloat tex[2];
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
  // Create color fragment shader, load and compile
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
  // Link Shaders together to a program
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

  GLubyte textureData[] =
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
    {{ -1.0f, 0.0f,  1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{ -1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{  1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {1.0f, 0.0f}},
    {{ -1.0f, 0.0f,  1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{  1.0f, 0.0f, -1.0f}, {1.0f, 0.2f, 0.0f, 1.0f}, {1.0f, 0.0f}},
    {{  1.0f, 0.0f,  1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}
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
    glm::vec3(0, 1, 0), // and looks at the origin
    glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
  );

  // Model matrix : an identity matrix (model will be at the origin)
  glm::mat4 triangleModelMat = glm::mat4(1.0f);
  glm::mat4 groundModelMat = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
  // Our ModelViewProjection : multiplication of our 3 matrices
  glm::mat4 triangleMvpMat = projectionMat * viewMat * triangleModelMat; // Remember, matrix multiplication is the other way around
  glm::mat4 groundMvpMat = projectionMat * viewMat * groundModelMat;
  // Get MVP Handle from Program
  GLuint triangleMatrixID = glGetUniformLocation(triangleShaderProgram, "MVP");
  GLuint groundMatrixID = glGetUniformLocation(groundShaderProgram, "MVP");
  glUseProgram(groundShaderProgram);
  GLuint bla = glGetUniformLocation(groundShaderProgram, "mytexture");
  glUniform1i(bla, 0); //TODO
  
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
    glBindVertexArray(triangleVAO);
    // activate the shaders
    glUseProgram(triangleShaderProgram);
    // update the MVP in the currently activated shader
    glUniformMatrix4fv(triangleMatrixID, 1, GL_FALSE, glm::value_ptr(triangleMvpMat));
    // draw the vertices as single triangles according to program
    glDrawArrays(GL_TRIANGLES, 0, 3);
    
    // draw Ground
    glBindVertexArray(groundVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, groundTexture);
    glUseProgram(groundShaderProgram);
    glUniformMatrix4fv(groundMatrixID, 1, GL_FALSE, glm::value_ptr(groundMvpMat));
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
  // OpenGL is reference counted and terminated by GLFW
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
