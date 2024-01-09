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
      realT val = minL[cur] + costSum[offset + cur];
      int p = pL[cur];
      if (minL[cur] + costSum[offset + cur] > minR[len - 1 - cur] - costSum[offset + cur])
      {
        val = minR[len - 1 - cur] - costSum[offset + cur];
        p = pR[len - 1 - cur];
      }
      if (val < dist[offset + cur] && cur != p)
      {
        dist[offset + cur] = val;
        prev[offset + cur] = offset + p;
      }
    }
    __syncthreads();
  }
}

__global__ static void sweepVia(realT *dist, int *prev, const realT *viaCost, int offsetX, int offsetY,
                                int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lenX || y >= lenY)
    return;
  x += offsetX;
  y += offsetY;

  int p[MAX_NUM_LAYER];
  for (int z = 0; z < LAYER; z++)
    p[z] = z;
  for (int z = 1; z < LAYER; z++)
  {
    if (dist[xyzToIdx(x, y, z, DIRECTION, N)] > dist[xyzToIdx(x, y, z - 1, DIRECTION, N)] + viaCost[xyzToIdx(x, y, z, DIRECTION, N)])
    {
      dist[xyzToIdx(x, y, z, DIRECTION, N)] = dist[xyzToIdx(x, y, z - 1, DIRECTION, N)] + viaCost[xyzToIdx(x, y, z, DIRECTION, N)];
      p[z] = p[z - 1];
    }
    if (dist[xyzToIdx(x, y, LAYER - 1 - z, DIRECTION, N)] > dist[xyzToIdx(x, y, LAYER - z, DIRECTION, N)] + viaCost[xyzToIdx(x, y, LAYER - z, DIRECTION, N)])
    {
      dist[xyzToIdx(x, y, LAYER - 1 - z, DIRECTION, N)] = dist[xyzToIdx(x, y, LAYER - z, DIRECTION, N)] + viaCost[xyzToIdx(x, y, LAYER - z, DIRECTION, N)];
      p[LAYER - 1 - z] = p[LAYER - z];
    }
  }
  for (int z = 0; z < LAYER; z++)
  {
    if (p[z] != z)
      prev[xyzToIdx(x, y, z, DIRECTION, N)] = xyzToIdx(x, y, p[z], DIRECTION, N);
  }
}

__global__ static void cleanDistPrev(realT *dist, int *prev, const int *mark, int offsetX, int offsetY,
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
  prev[idx] = idx;
  dist[idx] = mark[idx] ? 0.f : INFINITY_DISTANCE;
}

__global__ static void setRootPin(int *mark, int *isRoutedPin, const int *pinIndices, int numPins)
{
  for (int i = 0; i < numPins; i++)
    isRoutedPin[i] = 0;
  isRoutedPin[0] = 1;
  mark[pinIndices[0]] = 1;
}

// 回溯一条路径
__global__ static void tracePath(int *mark, int *isRoutedPin, int *routes, const realT *dist, const int *prev, const int *pinIndices, int numPins, int DIRECTION, int N)
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
  while (!mark[idx])
  {
    int prevIdx = prev[idx];
    auto [prevX, prevY, prevZ] = idxToXYZ(prevIdx, DIRECTION, N);
    if (z == prevZ) // wire
    {
      int startIdx = min(idx, prevIdx);
      int endIdx = max(idx, prevIdx);
      routes[++routes[0]] = startIdx;
      routes[++routes[0]] = endIdx;
      for (int tmpIdx = startIdx; tmpIdx <= endIdx; tmpIdx++)
      {
        if (tmpIdx != prevIdx)
          mark[tmpIdx] = 1;
      }
    }
    else // via
    {
      int startLayer = min(z, prevZ);
      int endLayer = max(z, prevZ);
      routes[++routes[0]] = min(idx, prevIdx);
      routes[++routes[0]] = max(idx, prevIdx);
      for (int l = startLayer; l <= endLayer; l++)
      {
        int tmpIdx = xyzToIdx(x, y, l, DIRECTION, N);
        if (tmpIdx != prevIdx)
          mark[tmpIdx] = 1;
      }
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
  devPrev = cuda_make_unique<int[]>(LAYER * N * N);
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

void BasicGamer::route(const std::vector<int> &pinIndices, int sweepTurns, const utils::BoxT<int> &box)
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
  cleanDistPrev<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devDist.get(), devPrev.get(), devMark.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
  for (int iter = 1; iter < pinIndices.size(); iter++)
  {
    for (int turn = 0; turn < sweepTurns; turn++)
    {
      sweepVia<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, 1), dim3(32, 32, 1)>>>(
          devDist.get(), devPrev.get(), devViaCost.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
      sweepWire<<<maxlen, std::min(1024, maxlen / 2), maxlen * (2 * sizeof(realT) + 2 * sizeof(int))>>>(
          devDist.get(), devPrev.get(), devWireCostSum.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
    }
    tracePath<<<1, 1>>>(devMark.get(), devIsRoutedPin.get(), devRoutes.get(), devDist.get(), devPrev.get(), devPinIndices.get(), numPins, DIRECTION, N);
    cleanDistPrev<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
        devDist.get(), devPrev.get(), devMark.get(), offsetX, offsetY, lenX, lenY, DIRECTION, N, X, Y, LAYER);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}
#endif