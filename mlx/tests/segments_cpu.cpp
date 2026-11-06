#include "engine/engine_cpu.h"
#include "toy/segments_main.h"
#include "mlx_kernels.h"
int main(int argc, char **argv) { return blk::segments_main<blk::EngineCpu>(argc, argv); }
