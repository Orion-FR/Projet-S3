#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "rotate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Image* loadImage(const char* filePath) 
{
    Image* img = (Image*)malloc(sizeof(Image));
    if (!img) {
        fprintf(stderr, "Erreur: Echec d'allocation de mémoire pour l'image.\n");
        return NULL;
    }
    
    img->data = stbi_load(filePath, &img->width, &img->height, &img->channels, 0);
    if (!img->data) {
        fprintf(stderr, "Erreur: Impossible de charger l'image %s\n", filePath);
        free(img);
        return NULL;
    }
    
    return img;
}

void freeImage(Image* img) {
    if (img) {
        if (img->data) {
            stbi_image_free(img->data);
        }
        free(img);
    }
}

int saveImage(const char* filePath, Image* img, int format) {
    int success = 0;
    if (format == stbi__png) 
    {
        success = stbi_write_png(filePath, img->width, img->height, img->channels, img->data, img->width * img->channels);
    } 
    else if (format == stbi__jpg)
    {
        success = stbi_write_jpg(filePath, img->width, img->height, img->channels, img->data, 100);
    } 
    else if (format == stbi__bmp) 
    {
        success = stbi_write_bmp(filePath, img->width, img->height, img->channels, img->data);
    }
    return success;
}

Image* rotateImage(const Image* img, float degrees) 
{
    if (!img || !img->data) 
    {
        return NULL;
    }

    float radians = degrees * M_PI / 180.0;

    int newWidth = (int)(abs(img->width * cos(radians)) + abs(img->height * sin(radians)));
    int newHeight = (int)(abs(img->width * sin(radians)) + abs(img->height * cos(radians)));

    Image* rotatedImg = (Image*)malloc(sizeof(Image));
    if (!rotatedImg) 
    {
        fprintf(stderr, "Erreur: Echec d'allocation de mémoire pour l'image !\n");
        return NULL;
    }

    rotatedImg->width = newWidth;
    rotatedImg->height = newHeight;
    rotatedImg->channels = img->channels;
    rotatedImg->data = (unsigned char*)calloc(newWidth * newHeight * img->channels, sizeof(unsigned char));
    if (!rotatedImg->data) 
    {
        fprintf(stderr, "Erreur: Echec d'allocation de mémoire pour l'image !\n");
        free(rotatedImg);
        return NULL;
    }

    int cx = img->width / 2;
    int cy = img->height / 2;
    int ncx = newWidth / 2;
    int ncy = newHeight / 2;

    for (int y = 0; y < img->height; y++) 
    {
        for (int x = 0; x < img->width; x++) 
        {
            int newX = (int)(cos(radians) * (x - cx) - sin(radians) * (y - cy) + ncx);
            int newY = (int)(sin(radians) * (x - cx) + cos(radians) * (y - cy) + ncy);

            if (newX >= 0 && newX < newWidth && newY >= 0 && newY < newHeight) 
            {
                for (int c = 0; c < img->channels; c++)
                {
                    rotatedImg->data[(newY * newWidth + newX) * img->channels + c] = img->data[(y * img->width + x) * img->channels + c];
                }
            }
        }
    }
    return rotatedImg;
}

int main() 
{
    return 0;
}