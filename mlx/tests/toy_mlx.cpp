// The toy proof of blocks/toy/toy_main.h with the MLX engine as the prover's; the library's verifier checks it.
#include "engine_mlx.h"
#include "toy/toy_main.h"
int main(int argc, char **argv) { return blk::toy_main<blk::EngineMlx>(argc, argv); }
