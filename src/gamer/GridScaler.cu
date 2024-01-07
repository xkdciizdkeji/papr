#ifdef ENABLE_CUDA
#include "GridScaler.cuh"

// --------------------------
// cuda kernel
// --------------------------

__global__ static void calculateCoarseCost(realT *coarseWireCost, realT *coarseViaCost, const realT *wireCost, const realT *viaCost,
                                           int coarseOffsetX, int coarseOffsetY, int coarseLengthX, int coarseLengthY,
                                           int coarseN, int coarseX, int coarseY, int scaleX, int scaleY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= coarseLengthX || y >= coarseLengthY || z >= LAYER)
    return;
  x += coarseOffsetX;
  y += coarseOffsetY;
  int idx = xyzToIdx(x, y, z, DIRECTION, coarseN);

  // wire cost
  realT wc = 0.0f;
  const realT *val = nullptr;
  int outerScale = 0;
  int outerBound = 0;
  int innerScale = 0;
  int innerBound = 0;
  if ((z & 1) ^ DIRECTION)
  {
    val = x > 0 && z > 0 ? wireCost + z * N * N + (y * scaleY) * N + (x - 1) * scaleX : nullptr;
    outerScale = scaleY;
    outerBound = min(scaleY, Y - y * scaleY);
    innerScale = scaleX;
    innerBound = min(2 * scaleX, X - (x - 1) * scaleX);
  }
  else
  {
    val = y > 0 && z > 0 ? wireCost + z * N * N + (x * scaleX) * N + (y - 1) * scaleY : nullptr;
    outerScale = scaleX;
    outerBound = min(scaleX, X - x * scaleX);
    innerScale = scaleY;
    innerBound = min(2 * scaleY, Y - (y - 1) * scaleY);
  }
  if (val)
  {
    for (int r = 0; r < outerBound; r++)
    {
      val += r * N;
      realT acc = 0.f, res = 0.f, cnt = 0.f;
      for (int t = innerScale; t < innerBound; t++)
        acc += val[t];
      for (int t = innerBound - 1, ascend = 0; t >= innerScale; t--, ascend ^= 1)
      {
        res += acc;
        cnt += 1.0f;
        if (ascend)
        {
          for (int tt = 1; tt < innerScale; tt++)
          {
            acc -= val[tt];
            res += acc;
            cnt += 1.0f;
          }
        }
        else
        {
          for (int tt = innerScale - 1; tt > 0; tt--)
          {
            acc += val[tt];
            res += acc;
            cnt += 1.0f;
          }
        }
        acc -= val[x];
      }
      wc += res / cnt;
    }
    wc /= static_cast<realT>(outerBound);
  }
  coarseWireCost[idx] = wc;

  // via cost
  realT vc = 0.f;
  for (int xx = x * scaleX, xxe = min((x + 1) * scaleX, X); xx < xxe; xx++)
    for (int yy = y * scaleY, yye = min((y + 1) * scaleY, Y); yy < yye; yy++)
      vc += viaCost[xyzToIdx(xx, yy, z, DIRECTION, N)];
  coarseViaCost[idx] = vc / static_cast<realT>(min(scaleX, X - x * scaleX) * min(scaleY, Y - y * scaleY));
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

std::vector<int> GridScaler::calculateCoarsePinIndices(const std::vector<int> &pinIndices)
{
  std::vector<int> coarsePinIndices(pinIndices.size());
  std::transform(pinIndices.begin(), pinIndices.end(), coarsePinIndices.begin(), [&](int idx)
                 {
    auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
    return xyzToIdx(x / scaleX, y / scaleY, z, DIRECTION, coarseN); });
  return std::move(coarsePinIndices);
}

utils::BoxT<int> GridScaler::calculateCoarseBoudingBox(const utils::BoxT<int> &box)
{
  return utils::BoxT<int>(box.lx() / scaleX, box.ly() / scaleY, (box.hx() + scaleX - 1) / scaleY, (box.hy() + scaleY - 1) / scaleY);
}

void GridScaler::scale()
{
  scale(utils::BoxT<int>(0, 0, coarseX, coarseY));
}

void GridScaler::scale(const utils::BoxT<int> &coarseBox)
{
  int coarseOffsetX = coarseBox.lx();
  int coarseOffsetY = coarseBox.ly();
  int coarseLengthX = coarseBox.width();
  int coarseLengthY = coarseBox.height();
  calculateCoarseCost<<<dim3((coarseX + 31) / 32, (coarseY + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devCoarseWireCost.get(), devCoarseViaCost.get(), devWireCost.get(), devViaCost.get(),
      coarseOffsetX, coarseOffsetY, coarseLengthX, coarseLengthY,
      coarseN, coarseX, coarseY, scaleX, scaleY, DIRECTION, N, X, Y, LAYER);
  checkCudaErrors(cudaDeviceSynchronize());
}
#endif