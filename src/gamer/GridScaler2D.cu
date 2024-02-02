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

void GridScaler2D::scalePin2DIndices(std::vector<int> &coarsePinIndices, const std::vector<int> &pinIndices) const
{
  coarsePinIndices.clear();
  std::transform(pinIndices.begin(), pinIndices.end(), std::back_inserter(coarsePinIndices), [&](int idx)
                 { return ((idx / X) / scaleY) * coarseX + ((idx % X) / scaleX); });
  std::sort(coarsePinIndices.begin(), coarsePinIndices.end());
  auto last = std::unique(coarsePinIndices.begin(), coarsePinIndices.end());
  coarsePinIndices.erase(last, coarsePinIndices.end());
}

utils::BoxT<int> GridScaler2D::coarsenBoudingBox(const utils::BoxT<int> &box) const
{
  return utils::BoxT<int>(
      box.lx() / scaleX, box.ly() / scaleY,
      std::min(coarseX, box.hx() / scaleX + 1), std::min(coarseY, box.hy() / scaleY + 1));
}

utils::BoxT<int> GridScaler2D::finingBoundingBox(const utils::BoxT<int> &box) const
{
  return utils::BoxT<int>(
      box.lx() * scaleX, box.ly() * scaleY,
      std::min(X, box.hx() * scaleX), std::min(Y, box.hy() * scaleY));
}

void GridScaler2D::getGuideFromRoutes2D(std::vector<utils::BoxT<int>> &guide2D, const int *routes2D) const
{
  guide2D.clear();
  for (int i = 0; i < routes2D[0]; i += 2)
  {
    int startX = routes2D[1 + i] % coarseX, startY = routes2D[1 + i] / coarseX;
    int endX = routes2D[2 + i] % coarseX, endY = routes2D[2 + i] / coarseX;
    guide2D.emplace_back(startX * scaleX, startY * scaleY,
                         std::min(X, (endX + 1) * scaleX), std::min(Y, (endY + 1) * scaleY));
  }
  // TODO: merge boxes if possible
}

void GridScaler2D::scaleCost2D()
{
  scaleCost2D(utils::BoxT<int>(0, 0, coarseX, coarseY));
}

void GridScaler2D::scaleCost2D(const utils::BoxT<int> &coarseBox)
{
  coarsenCost2D<<<dim3((coarseBox.width() + 31) / 32, (coarseBox.height() + 31) / 32), dim3(32, 32)>>>(
      devCoarseCost2D.get(), devCost2D.get(), coarseBox.lx(), coarseBox.ly(),
      coarseBox.width(), coarseBox.height(), coarseX, coarseY, scaleX, scaleY, X, Y);
  checkCudaErrors(cudaDeviceSynchronize());
}
#endif