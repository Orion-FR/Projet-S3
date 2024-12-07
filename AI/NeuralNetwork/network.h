#ifndef NETWORK_H
#define NETWORK_H

#include "neural.h"

typedef struct {
    int num_layers;       // Nombre de couches
    int *layer_sizes;     // Nombre de neurones par couche
    Neuron ***layers;     // Tableau de couches (chaque couche contient des neurones)
} Network;

// Initialisation et libération du réseau
Network *create_network(int num_layers, int *layer_sizes);
void free_network(Network *network);

// Propagation avant
void forward_pass(Network *network, double *inputs);

#endif