#ifdef ENABLE_CUDA
#include "BasicGamer.cuh"
#include <numeric>

// --------------------------
// cuda kernel
// --------------------------

__global__ static void calculateWireCostSum(realT *wireCostSum, const realT *wireCost, int offsetX, int offsetY,
                                            int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  extern __shared__ realT sum[];
  for (int i = 1; i < LAYER; i++)
  {
    if (blockIdx.x >= ((i & 1) ^ DIRECTION ? lenY : lenX))
      continue;
    int offset = i * N * N + blockIdx.x * N + ((i & 1) ^ DIRECTION ? offsetY * N + offsetX : offsetX * N + offsetY);
    int len = (i & 1) ^ DIRECTION ? lenX : lenY;
    for (int cur = threadIdx.x; cur < len; cur += blockDim.x)
      sum[cur] = wireCost[offset + cur];
    __syncthreads();
    for (int d = 0; (1 << d) < len; d++)
    {
      for (int cur = threadIdx.x; cur < len / 2; cur += blockDim.x)
      {
        // int src = (1 << d) + (threadIdx.x >> d << (d + 1)) - 1;
        // int dst = (1 << d) + threadIdx.x + (threadIdx.x >> d << d);
        int dst = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
        int src = (dst >> d << d) - 1;
        if (dst < len)
          sum[dst] += sum[src];
      }
      __syncthreads();
    }
    for (int cur = threadIdx.x; cur < len; cur += blockDim.x)
      wireCostSum[offset + cur] = sum[cur];
    __syncthreads();
  }
}

__global__ static void sweepWire(realT *dist, int *prev, const realT *costSum, int offsetX, int offsetY,
                                 int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  extern __shared__ int shared[];
  realT *minL = (realT *)(shared);
  realT *minR = (realT *)(minL + max(lenX, lenY));
  int *pL = (int *)(minR + max(lenX, lenY));
  int *pR = (int *)(pL + max(lenX, lenY));

  // layer 0
  if (blockIdx.x < ((0 & 1) ^ DIRECTION ? lenY : lenX))
  {
    int offset = 0 * N * N + blockIdx.x * N + ((0 & 1) ^ DIRECTION ? offsetY * N + offsetX : offsetX * N + offsetY);
    int len = (0 & 1) ^ DIRECTION ? lenX : lenY;
    for (int cur = threadIdx.x; cur < len; cur += blockDim.x)
      prev[offset + cur] = offset + cur;
  }
  // layer 1+
  for (int i = 1; i < LAYER; i++)
  {
    if (blockIdx.x >= ((i & 1) ^ DIRECTION ? lenY : lenX))
      continue;
    int offset = i * N * N + blockIdx.x * N + ((i & 1) ^ DIRECTION ? offsetY * N + offsetX : offsetX * N + offsetY);
    int len = (i & 1) ^ DIRECTION ? lenX : lenY;

    for (int cur = threadIdx.x; cur < len; cur += blockDim.x)
    {
      minL[cur] = dist[offset + cur] - costSum[offset + cur];
      minR[len - 1 - cur] = dist[offset + cur] + costSum[offset + cur];
      pL[cur] = cur;
      pR[len - 1 - cur] = cur;
    }
    __syncthreads();
    for (int d = 0; (1 << d) < len; d++)
    {
      // 对于长度为N的数组，需要N/2个线程工作
      for (int cur = threadIdx.x; cur < len / 2; cur += blockDim.x)
      {
        // int src = (1 << d) + (threadIdx.x >> d << (d + 1)) - 1;
        // int dst = (1 << d) + threadIdx.x + (threadIdx.x >> d << d);
        int dst = (cur >> d << (d + 1) | (1 << d)) | (cur & ((1 << d) - 1));
        int src = (dst >> d << d) - 1;
        if (dst < len)
        {
          if (minL[dst] > minL[src])
          {
            minL[dst] = minL[src];
            pL[dst] = pL[src];
          }
          if (minR[dst] > minR[src])
          {
            minR[dst] = minR[src];
            pR[dst] = pR[src];
          }
        }
      }
      __syncthreads();
    }
    for (int cur = threadIdx.x; cur < len; cur += blockDim.x)
    {
      if (minL[cur] + costSum[offset + cur] > minR[len - 1 - cur] - costSum[offset + cur])
      {
        dist[offset + cur] = minR[len - 1 - cur] - costSum[offset + cur];
        prev[offset + cur] = offset + pR[len - 1 - cur];
      }
      else
      {
        dist[offset + cur] = minL[cur] + costSum[offset + cur];
        prev[offset + cur] = offset + pL[cur];
      }
    }
    __syncthreads();
  }
}

__global__ static void sweepVia(realT *dist, int *prev, const realT *nonStackViaCost, realT unitViaCost,
                                int offsetX, int offsetY, int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lenX || y >= lenY)
    return;
  x += offsetX;
  y += offsetY;

  int p[MAX_NUM_LAYER];
  realT d[MAX_NUM_LAYER];
  realT o[MAX_NUM_LAYER];
  realT v = unitViaCost;
  int n = LAYER;
  for (int z = 0; z < LAYER; z++)
  {
    p[z] = z;
    d[z] = dist[xyzToIdx(x, y, z, DIRECTION, N)];
    o[z] = nonStackViaCost[xyzToIdx(x, y, z, DIRECTION, N)];
  }
  for (int i = 1, s = d[0], t, g = d[n - 1], h; i < n; i++)
  {
    // ascend
    t = s;
    s = d[i];
    if (d[i] > t + v)
    {
      d[i] = t + v;
      p[i] = i - 1;
    }
    if (d[i] > d[i - 1] + v + o[i - 1])
    {
      d[i] = d[i - 1] + v + o[i - 1];
      p[i] = p[i - 1];
    }
    // descend
    h = g;
    g = d[n - i - 1];
    if (d[n - i - 1] > h + v)
    {
      d[n - i - 1] = h + v;
      p[n - i - 1] = n - i;
    }
    if (d[n - i - 1] > d[n - i] + v + o[n - i])
    {
      d[n - i - 1] = d[n - i] + v + o[n - i];
      p[n - i - 1] = p[n - i];
    }
  }
  for (int z = 0; z < LAYER; z++)
  {
    dist[xyzToIdx(x, y, z, DIRECTION, N)] = d[z];
    prev[xyzToIdx(x, y, z, DIRECTION, N)] = xyzToIdx(x, y, p[z], DIRECTION, N);
  }
}

__global__ static void cleanDist(realT *dist, const int *mark, int offsetX, int offsetY,
                                 int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= lenX || y >= lenY || z >= LAYER)
    return;
  x += offsetX;
  y += offsetY;

  int idx = xyzToIdx(x, y, z, DIRECTION, N);
  dist[idx] = mark[idx] ? 0.f : INFINITY_DISTANCE;
}

__global__ static void setRootPin(int *mark, int *isRoutedPin, const int *pinIndices, int numPins)
{
  for (int i = 0; i < numPins; i++)
    isRoutedPin[i] = 0;
  isRoutedPin[0] = 1;
  mark[pinIndices[0]] = 1;
}

__global__ static void tracePath(int *mark, int *isRoutedPin, int *routes, const realT *dist, const int *allPrev, const int *pinIndices,
                                 int numPins, int numTurns, int DIRECTION, int N, int X, int Y, int LAYER)
{
  realT minDist = INFINITY_DISTANCE;
  int pinId = -1, idx = -1;

  // 寻找距离最短的未被连线的pin
  for (int i = 0; i < numPins; i++)
  {
    if (!isRoutedPin[i])
    {
      int p = pinIndices[i];
      if (dist[p] < minDist)
      {
        minDist = dist[p];
        idx = p;
        pinId = i;
      }
    }
  }
  if (pinId == -1)
    return;
  isRoutedPin[pinId] = 1;

  // backtracing
  auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
  for (int t = numTurns - 1; t >= 0; t--)
  {
    int prevIdx = allPrev[t * LAYER * N * N + idx];
    if (prevIdx == idx)
      continue;
    auto [prevX, prevY, prevZ] = idxToXYZ(prevIdx, DIRECTION, N);
    int startIdx = min(idx, prevIdx);
    int endIdx = max(idx, prevIdx);
    routes[++routes[0]] = startIdx;
    routes[++routes[0]] = endIdx;
    if (z == prevZ) // wire
    {
      for (int tmpIdx = startIdx; tmpIdx <= endIdx; tmpIdx++)
        mark[tmpIdx] = 1;
    }
    else // via
    {
      for (int l = min(z, prevZ), le = max(z, prevZ); l <= le; l++)
        mark[xyzToIdx(x, y, l, DIRECTION, N)] = 1;
    }
    idx = prevIdx;
    x = prevX;
    y = prevY;
    z = prevZ;
  }
}

// --------------------------
// GPUScaledRouter
// --------------------------

BasicGamer::BasicGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins)
    : DIRECTION(DIRECTION), N(N), X(X), Y(Y), LAYER(LAYER), numPins(0), maxNumPins(maxNumPins)
{
  devWireCostSum = cuda_make_unique<realT[]>(LAYER * N * N);
  devDist = cuda_make_unique<realT[]>(LAYER * N * N);
  devAllPrev = cuda_make_unique<int[]>(MAX_NUM_TURNS * LAYER * N * N);
  devMark = cuda_make_unique<int[]>(LAYER * N * N);

  devRoutes = cuda_make_shared<int[]>(maxNumPins * MAX_ROUTE_LEN_PER_PIN);
  devIsRoutedPin = cuda_make_unique<int[]>(maxNumPins);
  devPinIndices = cuda_make_unique<int[]>(maxNumPins);

  checkCudaErrors(cudaMemset(devRoutes.get(), 0, maxNumPins * MAX_ROUTE_LEN_PER_PIN));
}

bool BasicGamer::getIsRouted() const
{
  std::vector<int> isRoutePin(numPins);
  checkCudaErrors(cudaMemcpy(isRoutePin.data(), devIsRoutedPin.get(), numPins * sizeof(int), cudaMemcpyDeviceToHost));
  return std::reduce(isRoutePin.begin(), isRoutePin.end(), 1, [](int x, int y)
                     { return x & y; });
}

void BasicGamer::route(const std::vector<int> &pinIndices, int sweepTurns)
{
  route(pinIndices, sweepTurns, utils::BoxT<int>(0, 0, X, Y));
}

void BasicGamer::route(const std::vector<int> &pinIndices, int numTurns, const utils::BoxT<int> &box)
{
  int offsetX = box.lx();
  int offsetY = box.ly();
  int lenX = box.width();
  int lenY = box.height();
  int maxlen = std::max(lenX, lenY);

  numPins = static_cast<int>(pinIndices.size());
  checkCudaErrors(cudaMemset(devIsRoutedPin.get(), 0, numPins * sizeof(int)));
  checkCudaErrors(cudaMemcpy(devPinIndices.get(), pinIndices.data(), numPins * sizeof(int), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemset(devRoutes.get(), 0, maxNumPins * MAX_ROUTE_LEN_PER_PIN * sizeof(int)));
  checkCudaErrors(cudaMemset(devMark.get(), 0, LAYER * N * N * sizeof(int)));

  // wire cost sum
  calculateWireCostSum<<<maxlen, maxlen / 2, maxlen * sizeof(realT)>>>(
      devWireCostSum.get(), devWireCost.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
  // maze routing
  setRootPin<<<1, 1>>>(devMark.get(), devIsRoutedPin.get(), devPinIndices.get(), numPins);
  cleanDist<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devDist.get(), devMark.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
  for (int iter = 1; iter < pinIndices.size(); iter++)
  {
    for (int turn = 0; turn < numTurns; turn++)
    {
      if (turn & 1)
        sweepWire<<<maxlen, std::min(1024, maxlen / 2), maxlen * (2 * sizeof(realT) + 2 * sizeof(int))>>>(
            devDist.get(), devAllPrev.get() + turn * LAYER * N * N, devWireCostSum.get(),
            offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
      else
        sweepVia<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, 1), dim3(32, 32, 1)>>>(
            devDist.get(), devAllPrev.get() + turn * LAYER * N * N, devNonStackViaCost.get(), unitViaCost,
            offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
    }
    tracePath<<<1, 1>>>(
        devMark.get(), devIsRoutedPin.get(), devRoutes.get(), devDist.get(), devAllPrev.get(), devPinIndices.get(),
        numPins, numTurns, DIRECTION, N, X, Y, LAYER);
    cleanDist<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
        devDist.get(), devMark.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}
#endif