#include <stdio.h>
#include "genann.h"

int main(int argc, char *argv[])
{
    printf("GENANN example 1.\n");
    printf("Train a small ANN to the XOR function using backpropagation.\n");

    /* Input and expected out data for the XOR function. */
    const real_t input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const real_t output[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(2, 1, 2, 1);

    /* Train on the four labeled data points many times. */
    for (i = 0; i < 300; ++i) {
        genann_train(ann, input[0], output + 0, 3);
        genann_train(ann, input[1], output + 1, 3);
        genann_train(ann, input[2], output + 2, 3);
        genann_train(ann, input[3], output + 3, 3);
    }

    /* Run the network and see what it predicts. */
    #define SHOW_RESULT(n) printf("Output for [%1.f, %1.f] is %1.f.\n", input[n][0], input[n][1], *genann_run(ann, input[n]));
    SHOW_RESULT(0);
    SHOW_RESULT(1);
    SHOW_RESULT(2);
    SHOW_RESULT(3);
    #undef SHOW_RESULT

    genann_free(ann);
    return 0;
}
