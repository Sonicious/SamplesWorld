#version 410 core

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec2 vTexCoord;

layout(location = 0) out vec4 color;
layout(location = 1) out vec2 texCoord;

uniform mat4 MVP; // constant which must be uploaded from CPU

void main()
{
  gl_Position = MVP * vec4(vPos.x, vPos.y, vPos.z, 1.0);
  color = vColor;
  texCoord = vTexCoord;
}
