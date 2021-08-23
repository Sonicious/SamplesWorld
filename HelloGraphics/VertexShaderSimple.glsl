#version 410 core
layout (location = 0) in vec3 vPos; // must be idx of GL_VERTEX_POSITION_ATTRIBUTE_IDX
layout (location = 1) in vec4 vColor; // must be idx of GL_VERTEX_COLOR_ATTRIBUTE_IDX
out vec4 fColor;
uniform mat4 MVP; // constant which must be uploaded from CPU

void main()\n
{
  gl_Position = MVP * vec4(vPos.x, vPos.y, vPos.z, 1.0);
  fColor = vColor;
}
