#ifndef NEURON_H
#define NEURON_H

typedef struct {
    double *weights;
    double bias;
    double output;
} Neuron;


Neuron *create_neuron(int num_inputs);
void free_neuron(Neuron *neuron);

#endif