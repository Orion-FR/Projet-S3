#include <SDL.h>
#include "network.h"

void load_image_as_input(const char *path, double *inputs, int width, int height) {
    SDL_Surface *image = SDL_LoadBMP(path);
    if (!image) {
        printf("Erreur : %s\n", SDL_GetError());
        return;
    }

    Uint8 r, g, b;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Uint32 pixel = ((Uint32 *)image->pixels)[y * image->w + x];
            SDL_GetRGB(pixel, image->format, &r, &g, &b);
            inputs[y * width + x] = r / 255.0; // Normalisation
        }
    }
    SDL_FreeSurface(image);
}
