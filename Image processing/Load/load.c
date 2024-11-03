#include "load.h"
#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>

Image* loadImage(const char* filePath)
    {
    Image* img = (Image*)malloc(sizeof(Image));
    if (img == NULL)
        {
        fprintf(stderr, "Failed to allocate memory for image !\n");
        return NULL;
        }
    img->data = stbi_load(filePath, &img->width, &img->height, &img->channels, 0);
    if (img->data == NULL)
        {
        fprintf(stderr, "Failed to load image from file : %s\n", filePath);
        free(img);
        return NULL;
        }
    return img;
    }



int main() 
{
    return 0;
}