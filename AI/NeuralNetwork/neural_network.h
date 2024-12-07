#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <pthread.h>
#include <err.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#define NUM_THREADS 8
#define NUM_SAMPLES 1000
#define BATCH_SIZE (NUM_SAMPLES / NUM_THREADS)

typedef struct Neuron {
    unsigned int nbWeights;
    double *weights;

    double value;
    double bias;
    double delta;
} Neuron;

typedef struct Layer {
    unsigned int nbNeurons;
    Neuron *neurons;
} Layer;

typedef struct Network {
    unsigned int nbLayers;
    unsigned int sizeInput;
    unsigned int sizeHidden;
    unsigned int sizeOutput;
    Layer *layers;
} Network;

typedef struct {
    unsigned char r, g, b, a;
} RGBA;

typedef struct {
    int thread_id;
    Network* network;
    int* images;
    int* labels;
    int num_images;
    int num_pixels;
    int num_epochs;
    double learning_rate;
} ThreadData;

// ------ Neuron ------
Neuron newNeuron(unsigned int nbWeights);
void initNeuron(Neuron *neuron);
void freeNeuron(Neuron *neuron);

// ------ Layer ------
Layer newLayer(unsigned int sizeLayer, unsigned int sizePreviousLayer);
void freeLayer(Layer *layer);

// ------ Network ------
Network newNetwork(unsigned int sizeInput, unsigned int sizeHidden,
                   unsigned int nbHiddenLayers, unsigned int sizeOutput);
void initNetwork(Network *network);
void frontPropagation(Network *network, int input[]);
void freeNetwork(Network *network);
double backPropagation(Network *network, double expected[]);
void gradientDescent(Network *network, double learningRate);

// ------ Activation Functions ------
double sigmoid(double x);
double sigmoidPrime(double x);
double softmax(double x);
void softmaxLayer(Layer *layer);
double averageErrorRate(Network *network);

// ------ MNIST Data / Training ------
void loadMNISTImages(const char *filename, int *images, int num_images, int num_pixels);
void loadMNISTLabels(const char *filename, int *labels, int num_labels);
void trainOnMNIST(Network *network, int *images, int *labels, int num_images, int num_pixels, int num_epochs, double learning_rate);
void predict(Network *network, const char *image_filename, int num_pixels);

// ------ Image Processing ------
void readPGMImage(const char *filename, int *input, int num_pixels);
void readPNGImage(const char *filename, int *input, int num_pixels);

// ------ Saving datas ------
void saveNeural(const char *filename, Network *network);
void loadNeural(const char *filename, Network *network);

// ------ Main Function ------
int mainNeural(int train, char* path);

#endif