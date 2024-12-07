#include <stdlib.h>
#include "neural.h"

Neuron *create_neuron(int num_inputs) {
    Neuron *neuron = malloc(sizeof(Neuron));
    neuron->weights = malloc(num_inputs * sizeof(double));
    neuron->bias = 0.0;
    neuron->output = 0.0;
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double) rand() / RAND_MAX) - 0.5; // Entre -0.5 et 0.5
    }
    return neuron;
}

void free_neuron(Neuron *neuron) {
    free(neuron->weights);
    free(neuron);
}