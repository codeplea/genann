#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

const char *save_name = "example/xor.ann";

int main(int argc, char *argv[])
{
    printf("GENANN example 3.\n");
    printf("Load a saved ANN to solve the XOR function.\n");


    FILE *saved = fopen(save_name, "r");
    if (!saved) {
        printf("Couldn't open file: %s\n", save_name);
        exit(1);
    }

    genann *ann = genann_read(saved);
    fclose(saved);

    if (!ann) {
        printf("Error loading ANN from file: %s.", save_name);
        exit(1);
    }


    /* Input data for the XOR function. */
    const real_t input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

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
