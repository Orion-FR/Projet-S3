#include <stdlib.h>
#include <stdio.h>
#include "network.h"

Network *create_network(int num_layers, int *layer_sizes) {
    Network *network = malloc(sizeof(Network));
    network->num_layers = num_layers;
    network->layer_sizes = malloc(num_layers * sizeof(int));
    network->layers = malloc(num_layers * sizeof(Neuron **));

    for (int i = 0; i < num_layers; i++) {
        network->layer_sizes[i] = layer_sizes[i];
        network->layers[i] = malloc(layer_sizes[i] * sizeof(Neuron *));
        for (int j = 0; j < layer_sizes[i]; j++) {
            int num_inputs = (i == 0) ? 0 : layer_sizes[i - 1];
            network->layers[i][j] = create_neuron(num_inputs);
        }
    }
    return network;
}

void free_network(Network *network) {
    for (int i = 0; i < network->num_layers; i++) {
        for (int j = 0; j < network->layer_sizes[i]; j++) {
            free_neuron(network->layers[i][j]);
        }
        free(network->layers[i]);
    }
    free(network->layers);
    free(network->layer_sizes);
    free(network);
}

void forward_pass(Network *network, double *inputs) {
    // Propagation dans la première couche
    for (int i = 0; i < network->layer_sizes[0]; i++) {
        network->layers[0][i]->output = inputs[i];
    }

    // Propagation dans les couches suivantes
    for (int l = 1; l < network->num_layers; l++) {
        for (int j = 0; j < network->layer_sizes[l]; j++) {
            double sum = 0.0;
            for (int k = 0; k < network->layer_sizes[l - 1]; k++) {
                sum += network->layers[l - 1][k]->output * network->layers[l][j]->weights[k];
            }
            sum += network->layers[l][j]->bias;
            network->layers[l][j]->output = 1.0 / (1.0 + exp(-sum)); // Sigmoïde
        }
    }
}

void save_network(Network *network, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Erreur : Impossible de sauvegarder le réseau.\n");
        return;
    }

    for (int l = 1; l < network->num_layers; l++) { // Commence à la couche 1 (pas d'entrées pour la couche 0)
        for (int n = 0; n < network->layer_sizes[l]; n++) {
            Neuron *neuron = network->layers[l][n];
            for (int w = 0; w < network->layer_sizes[l - 1]; w++) {
                fprintf(file, "%lf ", neuron->weights[w]); // Poids
            }
            fprintf(file, "%lf\n", neuron->bias); // Biais
        }
    }

    fclose(file);
    printf("Réseau sauvegardé dans '%s'.\n", filename);
}

