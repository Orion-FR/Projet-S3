#include "greyscale.h"

void greyscale_image(Image* image) 
{
    for (int y = 0; y < image->height; ++y) 
    {
        for (int x = 0; x < image->width; ++x) 
        {
            Pixel* pixel = &image->pixels[y * image->width + x];
            unsigned char grey = (unsigned char)(0.3 * pixel->r + 0.59 * pixel->g + 0.11 * pixel->b);
            pixel->r = pixel->g = pixel->b = grey;
        }
    }
}