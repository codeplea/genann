#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "genann.h"

const char *save_name = "example/xor2.ann";
int main(int argc, char *argv[])
{
    printf("GENANN example 2.\n");
    printf("Train a small ANN to the XOR function using random search.\n");

    /* Input and expected out data for the XOR function. */
    const real_t input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const real_t output[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(2, 1, 2, 1);

    real_t err;
    real_t last_err = 1000;
    int count = 0;

    do {
        ++count;
        if (count % 1000 == 0) {
            /* We're stuck, start over. */
            genann_randomize(ann);
        }

        genann *save = genann_copy(ann);

        /* Take a random guess at the ANN weights. */
        for (i = 0; i < ann->total_weights; ++i) {
            ann->weight[i] += ((real_t)rand())/RAND_MAX-REAL_T_LITERAL(0.5);
        }

        /* See how we did. */
        err = 0;
	#define CALC_ERROR(x) pow(*genann_run(ann, input[x]) - output[x], REAL_T_LITERAL(2.0))
        err += CALC_ERROR(0);
        err += CALC_ERROR(1);
        err += CALC_ERROR(2);
        err += CALC_ERROR(3);
	#undef CALC_ERROR

        /* Keep these weights if they're an improvement. */
        if (err < last_err) {
            genann_free(save);
            last_err = err;
        } else {
            genann_free(ann);
            ann = save;
        }

    } while (err > REAL_T_LITERAL(0.01));

    printf("Finished in %d loops.\n", count);

    /* Run the network and see what it predicts. */
    #define SHOW_RESULT(n) printf("Output for [%1.f, %1.f] is %1.f.\n", input[n][0], input[n][1], *genann_run(ann, input[n]));
    SHOW_RESULT(0);
    SHOW_RESULT(1);
    SHOW_RESULT(2);
    SHOW_RESULT(3);
    #undef SHOW_RESULT

    FILE *saveto = fopen(save_name, "w");
    if (!saveto) {
        printf("Couldn't open file: %s\n", save_name);
        exit(1);
    }

    genann_write(ann, saveto);
    fclose(saveto);
    genann_free(ann);
    return 0;
}
