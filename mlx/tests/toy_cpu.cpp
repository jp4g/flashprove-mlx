// The same with the CPU engine, for comparing proofs bit by bit: equal seeds give equal digests.
#include "engine/engine_cpu.h"
#include "toy/toy_main.h"
#include "mlx_kernels.h"
int main(int argc, char **argv) { return blk::toy_main<blk::EngineCpu>(argc, argv); }
