#version 410 core

layout(location = 0) in vec4 fColor;
layout(location = 1) in vec2 texCoord;

out vec4 outColor;

void main()
{
  outColor = fColor;
  //outColor = vec4(1.0f, 1.0f, 1.0f, 0.5f);
}