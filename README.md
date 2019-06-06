[![Build Status](https://travis-ci.org/codeplea/genann.svg?branch=master)](https://travis-ci.org/codeplea/genann)

<img alt="Genann logo" src="https://codeplea.com/public/content/genann_logo.png" align="right" />

# Genann

Genann is a minimal, well-tested library for training and using feedforward
artificial neural networks (ANN) in C. Its primary focus is on being simple,
fast, reliable, and hackable. It achieves this by providing only the necessary
functions and little extra.

## Features

- **ANSI C with no dependencies**.
- Contained in a single source code and header file.
- Simple.
- Fast and thread-safe.
- Easily extendible.
- Implements backpropagation training.
- *Compatible with alternative training methods* (classic optimization, genetic algorithms, etc)
- Includes examples and test suite.
- Released under the zlib license - free for nearly any use.

## Building

Genann is self-contained in two files: `genann.c` and `genann.h`. To use Genann, you can simply add those two files to your project. It's written in standard C, so it's completely self-contained by default.

Genann can also use the canonical GNU Autotools buildsystem if you want to generate a shared library that you can install. For the default build, simply build it like any other project.

```bash
./configure
make
make install
```

Installing Genann will do two things. First, it will create a copy of the newly built library into your library directory. This is located at `/usr/local/lib` by default, and can be configured using `--prefix=/usr` as usual, or with whatever prefix you choose to use. Second, `genann.h` will be copied to your include directory, which by default is `/usr/local/include`, but can also be configured via the same `--prefix=PREFIX` option.

If you run `./configure --help`, you'll get a pretty good idea of the configuration options available. The original version exposed three `make` targets: `make linear`, `make sigmoid`, and `make threshold`, and those have been deprecated in lieu of configuration options. You can use these three by setting the `ACTIVATION_FUNCTION` variable when configuring the library, and you have the same choice between `SIGMOID`, `LINEAR`, and `THRESHOLD`.

## Example Configuration

```bash
./configuration --with-gmp --with-mpfr CC=gcc CFLAGS="-O3 -mtune=intel -march=skylake"
```

Note that the `--with-gmp` and `--with-mpfr` options, while they are functional in terms of the configuration and build system, the functionality needs to be added to the code itself, and for the moment are more proof of concept than anything else.

Running the above configuration should yield something like this, provided you have the requisite libraries installed, of course.

```bash
checking whether the C compiler works... yes
checking for C compiler default output file name... a.out
checking for suffix of executables... 
checking whether we are cross compiling... no
checking for suffix of object files... o
checking whether we are using the GNU C compiler... yes
checking whether gcc accepts -g... yes
checking for gcc option to accept ISO C89... none needed
checking for main in -lm... yes
checking for main in -lgmp... yes
checking for main in -lmpfr... yes
configure: creating ./config.status
config.status: creating Makefile
config.status: creating include/config.h
```

To build the library, simply run `make`.

```bash
gcc -O3 -mtune=intel -march=skylake -fPIC  -DHAVE_CONFIG_H  -I include -c -o genann.o src/genann.c
gcc -O3 -mtune=intel -march=skylake -fPIC -shared -o libgenann.so genann.o  -lm -lgmp -lmpfr 
```

 > <strong>Note:</strong> Remember that while the buildsystem <em>is</em> searching for, finding, and linking both GMP and MPFR, this functionality is experimental and for the moment only to demonstrate the extensibility of the build system. <em>Using these libraries at the moment yields absolutely no added functionality</em>.

Once `make` is finished (it shouldn't take long, it's only compiling one object file), there are several `make` targets at your disposal. <strong>Running `make check` is probably a really swell idea.</strong> Changing the activation function during the configuration can mess with the results, and at the moment it looks like it results in failing quite a few of the test cases. For this reason, it behooves you to to build the tests to make sure everything is working as expected.

If you would like to build the examples, simply run `make examples`. There are four of them, located in the `examples` directory, and this target will build them all in the project directory.

To remove build artifacts you can run `make clean`, `make clean-tests`, `make clean-examples`, or `make distclean`. The first will remove all object files and `persist.txt`, the second and third are self-explanatory, and the last will remove the configuration cache, `include/config.h`, `config.status`, and `config.log`. It will also run `make clean`, `make clean-tests`, and `make clean-examples` as well, so be sure you really want to delete everything before you do that.

If you do run `make distclean`, you'll have to re-run `./configure` to re-generate `include/config.h`.

## Installing

If you chose to build the library and you followed the building instructions above, simply run `make install` as root. Should you choose to uninstall the library, simply run `make uninstall` from the build directory, and the `genann.h` header and `libgenann.so` library will be promptly deleted, sent right back to the ether from whence they were came.

## Example Code

Four example programs are included with the source code.

- [`example1.c`](./example1.c) - Trains an ANN on the XOR function using backpropagation.
- [`example2.c`](./example2.c) - Trains an ANN on the XOR function using random search.
- [`example3.c`](./example3.c) - Loads and runs an ANN from a file.
- [`example4.c`](./example4.c) - Trains an ANN on the [IRIS data-set](https://archive.ics.uci.edu/ml/datasets/Iris) using backpropagation.

## Quick Example

We create an ANN taking 2 inputs, having 1 layer of 3 hidden neurons, and
providing 2 outputs. It has the following structure:

![NN Example Structure](./doc/e1.png)

We then train it on a set of labeled data using backpropagation and ask it to
predict on a test data point:

```C
#include "genann.h"

/* Not shown, loading your training and test data. */
double **training_data_input, **training_data_output, **test_data_input;

/* New network with 2 inputs,
 * 1 hidden layer of 3 neurons each,
 * and 2 outputs. */
genann *ann = genann_init(2, 1, 3, 2);

/* Learn on the training set. */
for (i = 0; i < 300; ++i) {
    for (j = 0; j < 100; ++j)
        genann_train(ann, training_data_input[j], training_data_output[j], 0.1);
}

/* Run the network and see what it predicts. */
double const *prediction = genann_run(ann, test_data_input[0]);
printf("Output for the first test data point is: %f, %f\n", prediction[0], prediction[1]);

genann_free(ann);
```

This example is to show API usage, it is not showing good machine learning
techniques. In a real application you would likely want to learn on the test
data in a random order. You would also want to monitor the learning to prevent
over-fitting.


## Usage

### Creating and Freeing ANNs
```C
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs);
genann *genann_copy(genann const *ann);
void genann_free(genann *ann);
```

Creating a new ANN is done with the `genann_init()` function. Its arguments
are the number of inputs, the number of hidden layers, the number of neurons in
each hidden layer, and the number of outputs. It returns a `genann` struct pointer.

Calling `genann_copy()` will create a deep-copy of an existing `genann` struct.

Call `genann_free()` when you're finished with an ANN returned by `genann_init()`.


### Training ANNs
```C
void genann_train(genann const *ann, double const *inputs,
        double const *desired_outputs, double learning_rate);
```

`genann_train()` will preform one update using standard backpropogation. It
should be called by passing in an array of inputs, an array of expected outputs,
and a learning rate. See *example1.c* for an example of learning with
backpropogation.

A primary design goal of Genann was to store all the network weights in one
contigious block of memory. This makes it easy and efficient to train the
network weights using direct-search numeric optimizion algorthims,
such as [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing),
[the Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm), [Simulated
Annealing](https://en.wikipedia.org/wiki/Simulated_annealing), etc.
These methods can be used by searching on the ANN's weights directly.
Every `genann` struct contains the members `int total_weights;` and
`double *weight;`.  `*weight` points to an array of `total_weights`
size which contains all weights used by the ANN. See *example2.c* for
an example of training using random hill climbing search.

### Saving and Loading ANNs

```C
genann *genann_read(FILE *in);
void genann_write(genann const *ann, FILE *out);
```
 
Genann provides the `genann_read()` and `genann_write()` functions for loading or saving an ANN in a text-based format.

### Evaluating

```C
double const *genann_run(genann const *ann, double const *inputs);
```

Call `genann_run()` on a trained ANN to run a feed-forward pass on a given set of inputs. `genann_run()`
will provide a pointer to the array of predicted outputs (of `ann->outputs` length).


## Hints

- All functions start with `genann_`.
- The code is simple. Dig in and change things.

## Extra Resources

The [comp.ai.neural-nets
FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/) is an excellent
resource for an introduction to artificial neural networks.

If you need an even smaller neural network library, check out the excellent single-hidden-layer library [tinn](https://github.com/glouw/tinn).

If you're looking for a heavier, more opinionated neural network library in C,
I recommend the [FANN library](http://leenissen.dk/fann/wp/). Another
good library is Peter van Rossum's [Lightweight Neural
Network](http://lwneuralnet.sourceforge.net/), which despite its name, is
heavier and has more features than Genann.

# Development

Should you choose to mess around with the build system, here are some tips from the man, the myth, the legend himself, on how the build system works. It's essentially just vanilla autotools, but if you haven't dabbled with it, it can feel like drinking documentation through a firehose.

Listen here, young grasshopper: if you edit `configure.ac`, you must re-run `autoconf`. After the first time you run `autoconf` proper, you can probably get by with simply running `autoreconf`, which runs a trimmed down version of `autoconf`, since you don't have to set everything everytime. That's what the cache is for, after all.

If you add additional libraries, required headers, or really anything that requires `config.h` to be up to speed, you have to re-run `autoheader ./configure.ac`. This parses the configuration file and adds any options to `config.h.in` that are then filled in when you actually run `./configure`. If this is confusing, welcome to the party.

This is what a header-involving modification requires:

```bash
autoheader ./configure.ac
autoconf
./configure
make
```

I also recommend running `make distclean` before re-running the configuration setup, since you can get pretty weird meta-build errors every once in a while.
