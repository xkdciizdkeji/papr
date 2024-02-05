#ifdef ENABLE_CUDA
#include "GridScaler2D.cuh"

// --------------------------
// cuda kernel
// --------------------------

__device__ static realT calculateCoarseCost(const realT *val, int scale, int bound)
{
  realT acc = 0.f, res = 0.f, cnt = 0.f;
  for (int t = scale; t < bound; t++)
    acc += val[t];
  for (int t = bound - 1, ascend = 0; t >= scale; t--, ascend ^= 1)
  {
    res += acc;
    cnt += 1.f;
    if (ascend)
    {
      for (int tt = 1; tt < scale; tt++)
      {
        acc -= val[tt];
        res += acc;
        cnt += 1.f;
      }
    }
    else
    {
      for (int tt = scale - 1; tt > 0; tt--)
      {
        acc += val[tt];
        res += acc;
        cnt += 1.f;
      }
    }
    acc -= val[t];
  }
  return res / cnt;
}

__global__ static void coarsenCost2D(realT *coarseCost2D, const realT *cost2D,
                                     int coarseOffsetX, int coarseOffsetY, int coarseLenX, int coarseLenY,
                                     int coarseX, int coarseY, int scaleX, int scaleY, int X, int Y)
{
  realT val[2 * MAX_SCALE];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= coarseLenX || y >= coarseLenY)
    return;
  x += coarseOffsetX;
  y += coarseOffsetY;

  // horizontal
  realT hori = 0.f;
  for (int r = 0, re = min(scaleY, Y - y * scaleY); r < re && x > 0; r++)
  {
    for (int t = 0, te = min(2 * scaleX, X - (x - 1) * scaleX); t < te; t++) // val = cost[(x - 1) * scaleX ~ x * scaleX, y * scaleY + r]
      val[t] = cost2D[(y * scaleY + r) * X + (x - 1) * scaleX + t];
    hori += calculateCoarseCost(val, scaleX, min(2 * scaleX, X - (x - 1) * scaleX));
  }
  coarseCost2D[y * coarseX + x] = hori / static_cast<realT>(min(scaleY, Y - y * scaleY));

  // vertical
  realT vert = 0.f;
  for (int r = 0, re = min(scaleX, X - x * scaleX); r < re && y > 0; r++)
  {
    for (int t = 0, te = min(2 * scaleY, Y - (y - 1) * scaleY); t < te; t++) // val = cost[x * scaleX + r, (y - 1) * scaleY ~ y * scaleY  ]
      val[t] = cost2D[X * Y + ((y - 1) * scaleY + t) * X + x * scaleX + r];
    vert += calculateCoarseCost(val, scaleY, min(2 * scaleY, Y - (y - 1) * scaleY));
  }
  coarseCost2D[coarseX * coarseY + y * coarseX + x] = vert / static_cast<realT>(min(scaleX, X - x * scaleX));
}

// -------------------------
// GridScaler2D
// -------------------------

GridScaler2D::GridScaler2D(int X, int Y, int scaleX, int scaleY)
    : X(X), Y(Y), scaleX(scaleX), scaleY(scaleY)
{
  coarseX = (X + scaleX - 1) / scaleX;
  coarseY = (Y + scaleY - 1) / scaleY;

  devCoarseCost2D = cuda_make_shared<realT[]>(2 * coarseX * coarseY);
}

void GridScaler2D::scale()
{
  scale(utils::BoxT<int>(0, 0, coarseX, coarseY));
}

void GridScaler2D::scale(const utils::BoxT<int> &coarseBox)
{
  coarsenCost2D<<<dim3((coarseBox.width() + 31) / 32, (coarseBox.height() + 31) / 32), dim3(32, 32)>>>(
      devCoarseCost2D.get(), devCost2D.get(), coarseBox.lx(), coarseBox.ly(),
      coarseBox.width(), coarseBox.height(), coarseX, coarseY, scaleX, scaleY, X, Y);
}
#endif