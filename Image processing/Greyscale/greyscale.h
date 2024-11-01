#ifndef GREYSCALE_H
#define GREYSCALE_H

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

typedef struct {
    int width;
    int height;
    Pixel* pixels;
} Image;

void greyscale_image(Image* image);

#endif