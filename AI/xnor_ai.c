#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//sigmoid function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//derivative of sigmoid function
double dx_sigmoid(double x) {
    return x * (1.0 - x);
}

//neural network able to learn XNOR
void train_xnor(double input_weights[2][2], double in_weights[2], double in_bias[2], double *out_bias, int nbocc) {
    // xnor inputs and expected outs
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected_outs[4] = {1, 0, 0, 1};

    // Train the network
    for (int e = 0; e < nbocc; e++) {
        for (int i = 0; i < 4; i++) {
            double input1 = inputs[i][0]; //Takes the first input of the xnor
            double input2 = inputs[i][1]; //Takes the second input of the xnor
            double expected_out = expected_outs[i]; //

            //Phase of "forward pass" (calculate the out)
            double in_input1 = sigmoid(input1 * input_weights[0][0] + input2 * input_weights[1][0] + in_bias[0]);
            double in_input2 = sigmoid(input1 * input_weights[0][1] + input2 * input_weights[1][1] + in_bias[1]);

            double out = sigmoid(in_input1 * in_weights[0] + in_input2 * in_weights[1] + *out_bias);

            //Phase of error calculation (calculate the error)
            double error = expected_out - out;

            //Phase of backpropagation (adjust weights and biases)
            double d_out = error * dx_sigmoid(out);
            
            double d_in1 = d_out * in_weights[0] * dx_sigmoid(in_input1);
            double d_in2 = d_out * in_weights[1] * dx_sigmoid(in_input2);

            //Update weights and biases 
            in_weights[0] += 0.1 * d_out * in_input1;
            in_weights[1] += 0.1 * d_out * in_input2;
            *out_bias += 0.1 * d_out;

            input_weights[0][0] += 0.1 * d_in1 * input1;
            input_weights[1][0] += 0.1 * d_in1 * input2;
            input_weights[0][1] += 0.1 * d_in2 * input1;
            input_weights[1][1] += 0.1 * d_in2 * input2;

            in_bias[0] += 0.1 * d_in1;
            in_bias[1] += 0.1 * d_in2;
        }
    }
}

void test_xnor(double input_weights[2][2], double in_weights[2], double in_bias[2], double out_bias, int nbocc) {
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    printf("Testing the neural network\n");
    printf("Expected outputs: 1 0 0 1\n");
    printf("Predicted outputs after %d attemps:\n\n", nbocc);
    for (int i = 0; i < 4; i++) {
        double input1 = inputs[i][0];
        double input2 = inputs[i][1];

        //Forward pass (use the trained weights and biases)
        double in_input1 = sigmoid(input1 * input_weights[0][0] + input2 * input_weights[1][0] + in_bias[0]);
        double in_input2 = sigmoid(input1 * input_weights[0][1] + input2 * input_weights[1][1] + in_bias[1]);

        double out = sigmoid(in_input1 * in_weights[0] + in_input2 * in_weights[1] + out_bias);

        printf("Input: (%d, %d) -> Predicted out: %.4f\n", (int)input1, (int)input2, out);
    }
}

int main(int argc, char** argv) {
    //Pseudo-random initialization of weights and biases
    double input_weights[2][2] = {{0.2, 0.4}, {0.6, 0.8}};
    double in_weights[2] = {0.5, 0.9};
    double in_bias[2] = {0.0, 0.0};
    double out_bias = 0.0;

    if (argc == 1) {
        train_xnor(input_weights, in_weights, in_bias, &out_bias, 50000);
        test_xnor(input_weights, in_weights, in_bias, out_bias, 50000);
        return 0;
    }
    else if (argc == 2)
    {
        train_xnor(input_weights, in_weights, in_bias, &out_bias, atoi(argv[1]));
        test_xnor(input_weights, in_weights, in_bias, out_bias, atoi(argv[1]));
        return 0;
    }
    else if (argc == 6)
    {
        if (atof(argv[2]) < 0 || atof(argv[2]) > 1 || atof(argv[3]) < 0 || atof(argv[3]) > 1 || atof(argv[4]) < 0 || atof(argv[4]) > 1 || atof(argv[5]) < 0 || atof(argv[5]) > 1)
        {
            printf("Input numbers must be between 0 and 1)\n");
            return 1;
        }
        double input_weights[2][2] = {{atof(argv[2]), atof(argv[3])}, {atof(argv[4]), atof(argv[5])}};
        train_xnor(input_weights, in_weights, in_bias, &out_bias, atoi(argv[1]));
        test_xnor(input_weights, in_weights, in_bias, out_bias, atoi(argv[1]));
        return 0;
    }
    else
    {
        printf("Usage: %s [nbocc] ( [input_weight0.0] [input_weight0.1] [input_weight1.0] [input_weight1.1] (input numbers must be between 0 and 1))\n", argv[0]);
        return 1;
    }
    return 0;
}