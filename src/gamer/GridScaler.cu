#ifdef ENABLE_CUDA

#include "GridScaler.cuh"

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


__global__ static void coarsenWireCost(realT *coarseWireCost, const realT *wireCost, int coarseOffsetX, int coarseOffsetY, int coarseLenX, int coarseLenY,
                                       int coarseN, int coarseX, int coarseY, int scaleX, int scaleY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= coarseLenX || y >= coarseLenY || z >= LAYER)
    return;
  x += coarseOffsetX;
  y += coarseOffsetY;

  // wire cost
  realT val[2 * MAX_SCALE];
  realT wc = 0.f;
  if ((z & 1) ^ DIRECTION) // YX
  {
    for (int r = 0, re = min(scaleY, Y - y * scaleY); r < re; r++)
    {
      for (int t = 0, te = min(2 * scaleX, X - (x - 1) * scaleX); t < te; t++)
        val[t] = wireCost[xyzToIdx((x - 1) * scaleX + t, y * scaleY + r, z, DIRECTION, N)];
      wc += calculateCoarseCost(val, scaleX, min(2 * scaleX, X - (x - 1) * scaleX));
    }
    wc /= static_cast<realT>(min(scaleY, Y - y * scaleY));
  }
  else
  {
    for (int r = 0, re = min(scaleX, X - x * scaleX); r < re; r++)
    {
      for (int t = 0, te = min(2 * scaleY, Y - (y - 1) * scaleY); t < te; t++)
        val[t] = wireCost[xyzToIdx(x * scaleX + r, (y - 1) * scaleY + t, z, DIRECTION, N)];
      wc += calculateCoarseCost(val, scaleY, min(2 * scaleY, Y - (y - 1) * scaleY));
    }
    wc /= static_cast<realT>(min(scaleX, X - x * scaleX));
  }
  coarseWireCost[xyzToIdx(x, y, z, DIRECTION, coarseN)] = wc;
}

__global__ static void coarsenViaCost(realT *coarseViaCost, const realT *viaCost, int coarseOffsetX, int coarseOffsetY, int coarseLenX, int coarseLenY,
                                      int coarseN, int coarseX, int coarseY, int scaleX, int scaleY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= coarseLenX || y >= coarseLenY || z >= LAYER)
    return;
  x += coarseOffsetX;
  y += coarseOffsetY;

  // via cost
  realT vc = 0.f;
  for (int xx = x * scaleX, xxe = min((x + 1) * scaleX, X); xx < xxe; xx++)
    for (int yy = y * scaleY, yye = min((y + 1) * scaleY, Y); yy < yye; yy++)
      vc += viaCost[xyzToIdx(xx, yy, z, DIRECTION, N)];
  coarseViaCost[xyzToIdx(x, y, z, DIRECTION, coarseN)] = vc / static_cast<realT>(min(scaleX, X - x * scaleX) * min(scaleY, Y - y * scaleY));
}

// --------------------------------
// GridScaler
// --------------------------------

GridScaler::GridScaler(int DIRECTION, int N, int X, int Y, int LAYER, int scaleX, int scaleY)
    : DIRECTION(DIRECTION), N(N), X(X), Y(Y), LAYER(LAYER), scaleX(scaleX), scaleY(scaleY)
{
  coarseX = (X + scaleX - 1) / scaleX;
  coarseY = (Y + scaleY - 1) / scaleY;
  coarseN = std::max(coarseX, coarseY);

  devCoarseWireCost = cuda_make_shared<realT[]>(LAYER * coarseN * coarseN);
  devCoarseViaCost = cuda_make_shared<realT[]>(LAYER * coarseN * coarseN);
}

void GridScaler::scale()
{
  scale(utils::BoxT<int>(0, 0, coarseX, coarseY));
}

void GridScaler::scale(const utils::BoxT<int> &coarseBox)
{
  coarsenWireCost<<<dim3((coarseBox.width() + 31) / 32, (coarseBox.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devCoarseWireCost.get(), devWireCost.get(),
      coarseBox.lx(), coarseBox.ly(), coarseBox.width(), coarseBox.height(),
      coarseN, coarseX, coarseY, scaleX, scaleY, DIRECTION, N, X, Y, LAYER);
  coarsenViaCost<<<dim3((coarseBox.width() + 31) / 32, (coarseBox.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devCoarseViaCost.get(), devViaCost.get(),
      coarseBox.lx(), coarseBox.ly(), coarseBox.width(), coarseBox.height(),
      coarseN, coarseX, coarseY, scaleX, scaleY, DIRECTION, N, X, Y, LAYER);
}

#endif