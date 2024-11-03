double sigmoid(double x); 

double dx_sigmoid(double x);

void train_xnor(double input_weights[2][2], double in_weights[2], double in_bias[2], double *out_bias, int nbocc);

void test_xnor(double input_weights[2][2], double in_weights[2], double in_bias[2], double out_bias, int nbocc);