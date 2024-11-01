#ifndef IMAGE_ROTATION_H
#define IMAGE_ROTATION_H

#include <stddef.h>
#include <stb_image.h>
#include <stb_image_write.h>

typedef struct 
{
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

Image* loadImage(const char* filePath);

void freeImage(Image* img);

int saveImage(const char* filePath, Image* img, int format);

Image* rotateImage(const Image* img, float degrees, RotationDirection direction);

#endif