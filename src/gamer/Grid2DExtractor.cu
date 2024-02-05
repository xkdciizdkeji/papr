#ifdef ENABLE_CUDA
#include "Grid2DExtractor.cuh"

// --------------------------
// Cuda kernel
// --------------------------

__global__ static void extract2DGrid(realT *cost2D, const realT *wireCost, int offsetX, int offsetY,
                                     int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lenX || y >= lenY)
    return;
  x += offsetX;
  y += offsetY;
  // horizontal
  realT hori = 0.f;
  // DIRECTION = 0 => layer 0 XY
  // DIRECTION = 1 => layer 0 YX
  for (int z = 1 + DIRECTION; z < LAYER; z += 2)
    hori += wireCost[xyzToIdx(x, y, z, DIRECTION, N)];
  cost2D[x + y * X] = hori / static_cast<realT>((LAYER - DIRECTION) / 2);
  // vertical
  realT vert = 0.f;
  for (int z = 1 + !DIRECTION; z < LAYER; z += 2)
    vert += wireCost[xyzToIdx(x, y, z, DIRECTION, N)];
  cost2D[X * Y + x + y * X] = vert / static_cast<realT>((LAYER - !DIRECTION) / 2);
}

// ----------------------------
// Grid2DExtractor
// ----------------------------

Grid2DExtractor::Grid2DExtractor(int DIRECTION, int N, int X, int Y, int LAYER)
    : DIRECTION(DIRECTION), N(N), X(X), Y(Y), LAYER(LAYER)
{
  devCost2D = cuda_make_shared<realT[]>(2 * X * Y);
}

void Grid2DExtractor::extract()
{
  extract(utils::BoxT<int>(0, 0, X, Y));
}

void Grid2DExtractor::extract(const utils::BoxT<int> &box)
{
  extract2DGrid<<<dim3((box.width() + 31) / 32, (box.height() + 31) / 32), dim3(32, 32)>>>(
      devCost2D.get(), devWireCost.get(), box.lx(), box.ly(), box.width(), box.height(), DIRECTION, N, X, Y, LAYER);
}
#endif