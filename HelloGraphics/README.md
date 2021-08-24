# OpenGL CUDA interaction examples

All Examples need OpenGL 4.1, CMAKE, glm, GLFW3

To show exactly the differences between the normal OpenGL and Cuda-OpenGL-interop it is recommended, to diff the two examples

In case of Surface and/or Texture memory please consult the documentation. The main difference is, that surface objects provides write-Operations into texture memory and texture objects provide hardware filtering and scaling.

## ModernOpenGL

This program is just an example, how modern OpenGl is working. There is an example of a static scene, which shows a small square with a texture and a triangle on top. 

This example is just the basis of the real examples.

## VerticesInteraction

This example give the idea, how to change the ModernOpenGL example to interact with Buffer Objects through CUDA. It exactly shows how a kernel interacts with a VertexBufferObject. The same procedure has to be made for uniform buffer objects.

## TextureInteractionPitched

Image Buffers in OpenGL are always Textures. The data-Type behind these are opaque and should not be interacted with directly. So one idea to deal with textures is to copy the data from texture memory into a 2D pitched memory bank in global memory. This procedure only makes sense when there is a read-modify-write procedure. In case of reading only, it is better to read from texture memory directly.

## TextureInteractionSurface

This example shows an alternative way to interact with textures from OpenGL. Here, a SurfaceObject is created, which gives the possibility to write into texture memory through simple and fast operations. 

