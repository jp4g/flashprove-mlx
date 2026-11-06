// Put first on the include path, this makes the library's own tests run on the MLX engine, unedited.
#pragma once
#include "engine_mlx.h"
namespace blk { using EngineCpu = EngineMlx; }
