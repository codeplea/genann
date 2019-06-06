#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

int main(int argc, char *argv[])
{
    printf("GENANN example 2.\n");
    printf("Train a small ANN to the XOR function using random search.\n");

    srand(time(0));

    /* Input and expected out data for the XOR function. */
    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double output[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(2, 1, 2, 1);

    double err;
    double last_err = 1000;
    int count = 0;

    do {
        ++count;
        if (count % 1000 == 0) {
            /* We're stuck, start over. */
            genann_randomize(ann);
            last_err = 1000;
        }

        genann *save = genann_copy(ann);

        /* Take a random guess at the ANN weights. */
        for (i = 0; i < ann->total_weights; ++i) {
            ann->weight[i] += ((double)rand())/RAND_MAX-0.5;
        }

        /* See how we did. */
        err = 0;
        err += pow(*genann_run(ann, input[0]) - output[0], 2.0);
        err += pow(*genann_run(ann, input[1]) - output[1], 2.0);
        err += pow(*genann_run(ann, input[2]) - output[2], 2.0);
        err += pow(*genann_run(ann, input[3]) - output[3], 2.0);

        /* Keep these weights if they're an improvement. */
        if (err < last_err) {
            genann_free(save);
            last_err = err;
        } else {
            genann_free(ann);
            ann = save;
        }

    } while (err > 0.01);

    printf("Finished in %d loops.\n", count);

    /* Run the network and see what it predicts. */
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[0][0], input[0][1], *genann_run(ann, input[0]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[1][0], input[1][1], *genann_run(ann, input[1]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[2][0], input[2][1], *genann_run(ann, input[2]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[3][0], input[3][1], *genann_run(ann, input[3]));

    genann_free(ann);
    return 0;
}
