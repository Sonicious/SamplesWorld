#version 410 core

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 texCoord;

out vec4 outColor;

void main()
{
  outColor = color;
}