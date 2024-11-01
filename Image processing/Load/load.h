#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H
#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>

typedef struct Image
{
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

Image* loadImage(const char* filePath);

void freeImage(Image* img);

#endif