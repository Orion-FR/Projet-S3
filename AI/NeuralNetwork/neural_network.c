#include "network.h"

// ------ Neuron ------

Neuron newNeuron(unsigned int nbWeights)
{
    Neuron neuron = {
        .nbWeights = nbWeights,
        .weights = NULL,
        .value = 0,
        .bias = 0,
        .delta = 0,
    };

    // Allocate memory for weights
    if (nbWeights != 0)
    {
        neuron.weights = malloc((nbWeights + 1) * sizeof(double));
        if (neuron.weights == NULL)
        {
            errx(EXIT_FAILURE, "Error while allocating memory");
        }
    }
    return neuron;
}

void* trainOnMNISTThread(void* thread_arg) {
    ThreadData* data = (ThreadData*)thread_arg;

    for (int epoch = 0; epoch < data->num_epochs; ++epoch) {
        double total_error = 0.0;

        for (int i = data->thread_id; i < data->num_images; i += NUM_THREADS) {
            // Prepare input vector
            int* input = malloc(data->num_pixels * sizeof(int));
            memcpy(input, &(data->images[i * data->num_pixels]), data->num_pixels * sizeof(int));

            // Forward propagation
            frontPropagation(data->network, input);

            // Prepare target output vector
            double target_output[10] = {0.0};
            target_output[data->labels[i]] = 1.0;

            // Backpropagation
            double error = backPropagation(data->network, target_output);
            total_error += error;

            // Gradient descent
            gradientDescent(data->network, data->learning_rate);

            free(input);
        }

        

        // Afficher l'erreur moyenne pour l'époque
        double average_error = total_error / data->num_images;
        printf("Thread %d, Epoch %d, Average Error: %f\n", data->thread_id, epoch + 1, average_error);
    }

    pthread_exit(NULL);
}

// Should only be called in initNetwork because of : srand ( time ( NULL));
void initNeuron(Neuron *neuron)
{
    unsigned int nbWeights = neuron->nbWeights;
    for (unsigned int i = 0; i < nbWeights; i++)
    {
        neuron->weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    neuron->bias = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

void freeNeuron(Neuron *neuron)
{
    free(neuron->weights);
}


// ------ Layer ------

Layer newLayer(unsigned int size, unsigned int sizePreviousLayer)
{
    Layer layer = { .nbNeurons = size, .neurons = NULL };

    // Allocate memory for neurons, calloc already put the + 1 for the \0
    layer.neurons = malloc((size + 1) * sizeof(struct Neuron));
    if (layer.neurons == NULL)
    {
        errx(EXIT_FAILURE, "Error while allocating memory");
    }

    // Create all the neurons depending on the size of the previous layer
    for (unsigned int i = 0; i < size; i++)
    {
        layer.neurons[i] = newNeuron(sizePreviousLayer);
    }

    return layer;
}

void freeLayer(Layer *layer)
{
    for (unsigned int i = 0; i < layer->nbNeurons; i++)
    {
        Neuron *neuron = &(layer->neurons[i]);
        freeNeuron(neuron);
    }
    free(layer->neurons);
}


// ------ Network ------

Network newNetwork(unsigned int sizeInput, unsigned int sizeHidden,
                   unsigned int nbHiddenLayers, unsigned int sizeOutput)
{
    Network network = { .nbLayers =
                            nbHiddenLayers + 2, // Add input and output layer
                        .sizeInput = sizeInput,
                        .sizeHidden = sizeHidden,
                        .sizeOutput = sizeOutput,
                        .layers = NULL };

    // Allocate memory for all layers
    network.layers = malloc((network.nbLayers + 1) * sizeof(struct Layer));
    if (network.layers == NULL)
    {
        errx(EXIT_FAILURE, "Error while allocating memory");
    }

    // Create the input layer
    network.layers[0] = newLayer(sizeInput, 0);

    // Create all hidden layer with the nbNeurons of the previous one
    for (unsigned int i = 1; i < network.nbLayers - 1; i++)
    {
        network.layers[i] =
            newLayer(sizeHidden, network.layers[i - 1].nbNeurons);
    }

    // Create the ouput layer
    network.layers[network.nbLayers - 1] =
        newLayer(sizeOutput, network.layers[network.nbLayers - 2].nbNeurons);

    return network;
}

// Initialize neural network
void initNetwork(Network *network)
{
    srand(time(NULL));
    unsigned int nbLayers = network->nbLayers;
    unsigned int nbNeurons;

    for (unsigned int i = 0; i < nbLayers; i++)
    {
        Layer *layer = &(network->layers[i]);
        nbNeurons = layer->nbNeurons;
        for (unsigned int j = 0; j < nbNeurons; j++)
        {
            initNeuron(&(layer->neurons[j]));
        }
    }
}

void frontPropagation(Network *network, int input[])
{
    // First layer
    Layer *layer = &(network->layers[0]);
    for (unsigned int i = 0; i < layer->nbNeurons; i++)
    {
        layer->neurons[i].value = input[i];
    }

    // Hiddens layer
    unsigned int nbLayers = network->nbLayers;
    unsigned int nbNeurons;
    // For each layer
    for (unsigned int i = 1; i < nbLayers; i++)
    {
        Layer prevLayer = network->layers[i - 1];
        layer = &(network->layers[i]);
        nbNeurons = layer->nbNeurons;

        // For each neuron of the actual layer
        for (unsigned int j = 0; j < nbNeurons; j++)
        {
            Neuron *neuron = &(layer->neurons[j]);
            double sum = 0.0;

            // Calcul new neuron value based on his weights and the value of
            // previous layer
            for (unsigned int k = 0; k <= prevLayer.nbNeurons; k++)
            {
                sum += neuron->weights[k] * prevLayer.neurons[k].value;
            }
            // sum += neuron->bias;
            layer->neurons[j].value = sigmoid(sum);
        }
    }

    // Output layer
    // Layer prevLayer = network->layers[network->nbLayers - 2];
    layer = &(network->layers[network->nbLayers - 1]);
    softmaxLayer(layer);
   
}

void freeNetwork(Network *network)
{
    for (unsigned int i = 0; i < network->nbLayers; i++)
    {
        Layer *layer = &(network->layers[i]);
        freeLayer(layer);
    }
    free(network->layers);
}


double backPropagation(Network *network, double expected[])
{
    double errorRate = 0.0;
    double errorTemp = 0.0;

    unsigned int nbLayers = network->nbLayers;

    // Output layer
    Layer *outputLayer = &(network->layers[nbLayers - 1]);

    // NbNeurons of lastlayer and expected are equals
    for (unsigned int i = 0; i < outputLayer->nbNeurons; i++)
    {
        Neuron *neuron = &(outputLayer->neurons[i]);
        errorTemp = expected[i] - neuron->value;
        neuron->delta = errorTemp * sigmoidPrime(neuron->value);
        errorRate += (errorTemp * errorTemp);
    }

    // For all layer except the input
    for (unsigned int i = nbLayers - 1; i >= 2; i--)
    {
        Layer layer = network->layers[i];
        Layer *previousLayer =
            &(network->layers[i - 1]); // Modify weights of this layer
        // For each neurons
        for (unsigned int j = 0; j < previousLayer->nbNeurons; j++)
        {
            errorTemp = 0.0;
            Neuron *neuron = &(previousLayer->neurons[j]);
            // Calculate error rate based on all neuron in the next layer and
            // all weights of the actual neuron
            for (unsigned int k = 0; k < layer.nbNeurons; k++)
            {
                errorTemp +=
                    layer.neurons[k].delta * layer.neurons[k].weights[j];
            }
            neuron->delta = errorTemp * sigmoidPrime(neuron->value);
        }
    }
    return errorRate;
}

void gradientDescent(Network *network, double learningRate)
{
    // Gradient descent
    for (unsigned int i = network->nbLayers - 1; i >= 1; i--)
    {
        Layer *layer = &(network->layers[i]);
        Layer *previousLayer = &(network->layers[i - 1]);
        // For each neurons in the layer
        for (unsigned int j = 0; j < layer->nbNeurons; j++)
        {
            // For each neurons on the layer
            Neuron *neuron = &(layer->neurons[j]);
            for (unsigned int k = 0; k < previousLayer->nbNeurons; k++)
            {
                // For each weights on the neuron of the previous layer
                neuron->weights[k] += neuron->delta
                    * previousLayer->neurons[k].value * learningRate;

                neuron->bias += neuron->delta * previousLayer->neurons[k].value
                    * learningRate;
            }
        }
    }
}


// ------ Activation Functions ------

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidPrime(double x) {
    return x * (1 - x);
}


double softmax(double x)
{
    return exp(x) / (1 + exp(-x));
}

void softmaxLayer(Layer *layer)
{
    double max_val = layer->neurons[0].value;

    // Find the maximum value in the layer
    for (unsigned int i = 1; i < layer->nbNeurons; i++)
    {
        if (layer->neurons[i].value > max_val)
        {
            max_val = layer->neurons[i].value;
        }
    }

    double sum = 0.0;

    // Apply softmax with regularization
    for (unsigned int i = 0; i < layer->nbNeurons; i++)
    {
        layer->neurons[i].value = exp(layer->neurons[i].value - max_val);
        sum += layer->neurons[i].value;
    }

    for (unsigned int i = 0; i < layer->nbNeurons; i++)
    {
        layer->neurons[i].value /= sum;
    }
}


double averageErrorRate(Network *network)
{
    double average = 0.0;
    for (unsigned int i = 0; i < network->nbLayers; i++)
    {
        for (unsigned int j = 0; j < network->layers[i].nbNeurons; j++)
        {
            average += network->layers[i].neurons[j].delta;
        }
        average /= network->layers[i].nbNeurons;
    }

    return average * -1;
}

// ------ MNIST Data ------


// Function to load MNIST images
void loadMNISTImages(const char *filename, int *images, int num_images, int num_pixels) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 16, SEEK_SET);  // Skip header

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_pixels; ++j) {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, file);
            images[i * num_pixels + j] = (int)pixel;
        }
    }

    fclose(file);
}

// Function to load MNIST labels
void loadMNISTLabels(const char *filename, int *labels, int num_labels) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 8, SEEK_SET);  // Skip header

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, file);
        labels[i] = (int)label;
    }

    fclose(file);
}

// Function to train the neural network on MNIST data
void trainOnMNIST(Network *network, int *images, int *labels, int num_images, int num_pixels, int num_epochs, double learning_rate) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_error = 0.0;

        for (int i = 0; i < num_images; ++i) {
            // Prepare input vector
            int *input = malloc(num_pixels * sizeof(int));
            memcpy(input, &images[i * num_pixels], num_pixels * sizeof(int));
            // Forward propagation
            frontPropagation(network, input);

            // Prepare target output vector
            double target_output[10] = {0.0};
            target_output[labels[i]] = 1.0;

            // Backpropagation
            double error = backPropagation(network, target_output);
            total_error += error;
            // Gradient descent
            gradientDescent(network, learning_rate);

            free(input);
        }

        // Calculate and print average error for the epoch
        double average_error = total_error / num_images;
        printf("Epoch %d, Average Error: %f\n", epoch + 1, average_error);
    }
}

void predict(Network *network, const char *image_filename, int num_pixels) {
    int input[num_pixels];
    readPNGImage(image_filename, input, num_pixels);

    frontPropagation(network, input);

    // Print the raw output values
    printf("Raw Output Values:\n");
    unsigned int max = 0;
    for (unsigned int i = 0; i < network->sizeOutput; ++i) {
        if(network->layers[network->nbLayers - 1].neurons[i].value>max)
            max = network->layers[network->nbLayers - 1].neurons[i].value>max;
    }
    printf("%u\n", max);
}

// Fonction pour lire et prétraiter une image au format PGM
void readPGMImage(const char *filename, int *input, int num_pixels) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier image : %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Lire l'en-tête PGM
    char magic[3];
    int width, height, max_value;
    fscanf(file, "%2s %d %d %d", magic, &width, &height, &max_value);

    if (strcmp(magic, "P2") != 0) {
        fprintf(stderr, "Format PGM invalide dans le fichier : %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Redimensionner l'image à la taille d'entrée souhaitée
    int resized_width = sqrt(num_pixels);
    int resized_height = sqrt(num_pixels);

    // Lire les valeurs des pixels et les mettre à l'échelle pour qu'elles correspondent à la taille souhaitée
    for (int i = 0; i < resized_height; ++i) {
        for (int j = 0; j < resized_width; ++j) {
            int pixel_value;
            fscanf(file, "%d", &pixel_value);
            input[i * resized_width + j] = pixel_value * num_pixels / max_value;
        }
    }

    fclose(file);
}

void readPNGImage(const char *filename, int *input, int num_pixels) {
    if (IMG_Init(IMG_INIT_PNG) == 0) {
        fprintf(stderr, "Erreur lors de l'initialisation de SDL_image : %s\n", IMG_GetError());
        exit(EXIT_FAILURE);
    }

    // Load the image
    SDL_Surface *image = IMG_Load(filename);
    if (!image) {
        fprintf(stderr, "Erreur lors du chargement de l'image PNG : %s\n", IMG_GetError());
        exit(EXIT_FAILURE);
    }

    // Ensure the image is grayscale (1 byte per pixel)
    if (image->format->BytesPerPixel != 1) {
        fprintf(stderr, "L'image doit être en niveaux de gris (8 bits par pixel)\n");
        SDL_FreeSurface(image);
        exit(EXIT_FAILURE);
    }

    // Ensure the number of pixels matches
    int width = image->w;
    int height = image->h;
    if (width * height != num_pixels) {
        fprintf(stderr, "Le nombre de pixels (%d) ne correspond pas au nombre attendu (%d)\n", width * height, num_pixels);
        SDL_FreeSurface(image);
        exit(EXIT_FAILURE);
    }

    // Copy pixel data to the input array
    Uint8 *pixels = (Uint8 *)image->pixels;
    for (int i = 0; i < num_pixels; ++i) {
        input[i] = pixels[i];
    }

    // Clean up
    SDL_FreeSurface(image);
}

// Function to save neural network data to a file
void saveNeural(const char *filename, Network *network) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Save network parameters
    fwrite(&network->sizeInput, sizeof(unsigned int), 1, file);
    fwrite(&network->sizeHidden, sizeof(unsigned int), 1, file);
    fwrite(&network->nbLayers, sizeof(unsigned int), 1, file);
    fwrite(&network->sizeOutput, sizeof(unsigned int), 1, file);

    // Save each layer separately
    for (unsigned int i = 0; i < network->nbLayers; ++i) {
        Layer *layer = &(network->layers[i]);

        // Save layer parameters
        fwrite(&layer->nbNeurons, sizeof(unsigned int), 1, file);

        // Save each neuron separately
        for (unsigned int j = 0; j < layer->nbNeurons; ++j) {
            Neuron *neuron = &(layer->neurons[j]);

            // Save neuron parameters
            fwrite(&neuron->nbWeights, sizeof(unsigned int), 1, file);

            // Save neuron weights, value, bias, and delta
            fwrite(neuron->weights, sizeof(double), neuron->nbWeights, file);
            fwrite(&neuron->value, sizeof(double), 1, file);
            fwrite(&neuron->bias, sizeof(double), 1, file);
            fwrite(&neuron->delta, sizeof(double), 1, file);
        }
    }

    fclose(file);
}

// Function to load neural network data from a file
void loadNeural(const char *filename, Network *network) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read network parameters
    fread(&network->sizeInput, sizeof(unsigned int), 1, file);
    fread(&network->sizeHidden, sizeof(unsigned int), 1, file);
    fread(&network->nbLayers, sizeof(unsigned int), 1, file);
    fread(&network->sizeOutput, sizeof(unsigned int), 1, file);

    // Allocate memory for layers
    network->layers = malloc((network->nbLayers + 1) * sizeof(struct Layer));

    // Read each layer separately
    for (unsigned int i = 0; i < network->nbLayers; ++i) {
        Layer *layer = &(network->layers[i]);

        // Read layer parameters
        fread(&layer->nbNeurons, sizeof(unsigned int), 1, file);

        // Allocate memory for neurons
        layer->neurons = malloc((layer->nbNeurons + 1) * sizeof(struct Neuron));

        // Read each neuron separately
        for (unsigned int j = 0; j < layer->nbNeurons; ++j) {
            Neuron *neuron = &(layer->neurons[j]);

            // Read neuron parameters
            fread(&neuron->nbWeights, sizeof(unsigned int), 1, file);

            // Allocate memory for neuron weights
            neuron->weights = malloc((neuron->nbWeights + 1) * sizeof(double));

            // Read neuron weights, value, bias, and delta
            fread(neuron->weights, sizeof(double), neuron->nbWeights, file);
            fread(&neuron->value, sizeof(double), 1, file);
            fread(&neuron->bias, sizeof(double), 1, file);
            fread(&neuron->delta, sizeof(double), 1, file);
        }
    }

    fclose(file);
}

int mainNeural(int train, char* path) {
    

    // Assuming a fixed image size of 28x28
    int num_pixels = 28 * 28;

    if (train == 0) {
        // Créer et initialiser un nouveau réseau neuronal
        Network network = newNetwork(num_pixels, 128, 15, 10);
        initNetwork(&network);

        // Charger les données MNIST
        int num_images = 60000;  // Ajustez au besoin
        int* images = malloc(num_images * num_pixels * sizeof(int));
        int labels[num_images];
        loadMNISTImages("train-images.idx3-ubyte", images, num_images, num_pixels);
        loadMNISTLabels("train-labels.idx1-ubyte", labels, num_images);

        // Configuration des threads
        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];

        for (int i = 0; i < NUM_THREADS; ++i) {
            thread_data[i].thread_id = i;
            thread_data[i].network = &network;
            thread_data[i].images = images;
            thread_data[i].labels = labels;
            thread_data[i].num_images = num_images;
            thread_data[i].num_pixels = num_pixels;
            thread_data[i].num_epochs = 50;       
            thread_data[i].learning_rate = 0.01;  

            int result = pthread_create(&threads[i], NULL, trainOnMNISTThread, (void*)&thread_data[i]);
            if (result) {
                printf("Erreur lors de la création du thread %d, code d'erreur : %d\n", i, result);
                exit(EXIT_FAILURE);
            }
        }

        // Attendre que tous les threads aient terminé
        for (int i = 0; i < NUM_THREADS; ++i) {
            pthread_join(threads[i], NULL);
        }

        // Enregistrez le réseau neuronal formé
        saveNeural("saved_network.txt", &network);

        // Libérez la mémoire allouée
        freeNetwork(&network);
        free(images);

        printf("Entraînement terminé. Réseau neuronal sauvegardé.\n");
    }  else if (train == 1) {
        // Load saved neural network and predict the value of an image

        // Create a new neural network
        Network network = newNetwork(num_pixels, 128, 15, 10);
        initNetwork(&network);

        // Load the saved neural network
        loadNeural("saved_network.txt", &network);

        // Get the path to the image from the user

        // Predict the value of the provided image
        predict(&network, path, num_pixels);

        // Free allocated memory
        freeNetwork(&network);

    } else {
        fprintf(stderr, "Invalid argument. Use 'train' or 'use'.\n");
        exit(EXIT_FAILURE);
    }

    return 0;
}