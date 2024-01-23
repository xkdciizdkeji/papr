#ifdef ENABLE_CUDA
#include "BasicGamer2D.cuh"
#include <numeric>

// ---------------------------------
// cuda kernel
// ---------------------------------

__global__ static void calculateWireCostSumHorizontal(realT *costSum, const realT *cost,
                                                      int offsetX, int offsetY, int lenX, int lenY, int X, int Y)
{
  extern __shared__ realT sum[];
  int offset = (offsetY + blockIdx.x) * X + offsetX;
  for (int cur = threadIdx.x; cur < lenX; cur += blockDim.x)
    sum[cur] = cost[offset + cur];
  __syncthreads();
  for (int d = 0; (1 << d) < lenX; d++)
  {
    for (int cur = threadIdx.x; cur < lenX / 2; cur += blockDim.x)
    {
      int dst = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
      int src = (dst >> d << d) - 1;
      if (dst < lenX)
        sum[dst] += sum[src];
    }
    __syncthreads();
  }
  for (int cur = threadIdx.x; cur < lenX; cur += blockDim.x)
    costSum[offset + cur] = sum[cur];
}

__global__ static void calculateWireCostSumVertical(realT *costSum, const realT *cost,
                                                    int offsetX, int offsetY, int lenX, int lenY, int X, int Y)
{
  extern __shared__ realT sum[];
  int offset = offsetY * X + (offsetX + blockIdx.x);
  for (int cur = threadIdx.x; cur < lenY; cur += blockDim.x)
    sum[cur] = cost[offset + cur * X];
  __syncthreads();
  for (int d = 0; (1 << d) < lenY; d++)
  {
    for (int cur = threadIdx.x; cur < lenY / 2; cur += blockDim.x)
    {
      int dst = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
      int src = (dst >> d << d) - 1;
      if (dst < lenY)
        sum[dst] += sum[src];
    }
    __syncthreads();
  }
  for (int cur = threadIdx.x; cur < lenY; cur += blockDim.x)
    costSum[offset + cur * X] = sum[cur];
}

__global__ static void sweepWireHorizontal(realT *dist, int *prev, const realT *costSum,
                                           int offsetX, int offsetY, int lenX, int lenY, int X, int Y)
{
  extern __shared__ int shared[];
  realT *minL = (realT *)(shared);
  realT *minR = (realT *)(minL + lenX);
  int *pL = (int *)(minR + lenX);
  int *pR = (int *)(pL + lenX);

  int offset = (offsetY + blockIdx.x) * X + offsetX;
  for (int cur = threadIdx.x; cur < lenX; cur += blockDim.x)
  {
    minL[cur] = dist[offset + cur] - costSum[offset + cur];
    minR[lenX - 1 - cur] = dist[offset + cur] + costSum[offset + cur];
    pL[cur] = cur;
    pR[lenX - 1 - cur] = cur;
  }
  __syncthreads();
  for (int d = 0; (1 << d) < lenX; d++)
  {
    for (int cur = threadIdx.x; cur < lenX / 2; cur += blockDim.x)
    {
      int dst = (cur >> d << (d + 1) | (1 << d)) | (cur & ((1 << d) - 1));
      int src = (dst >> d << d) - 1;
      if (dst < lenX)
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
  for (int cur = threadIdx.x; cur < lenX; cur += blockDim.x)
  {
    if (minL[cur] + costSum[offset + cur] > minR[lenX - 1 - cur] - costSum[offset + cur])
    {
      dist[offset + cur] = minR[lenX - 1 - cur] - costSum[offset + cur];
      prev[offset + cur] = offset + pR[lenX - 1 - cur];
    }
    else
    {
      dist[offset + cur] = minL[cur] + costSum[offset + cur];
      prev[offset + cur] = offset + pL[cur];
    }
  }
}

__global__ static void sweepWireVertical(realT *dist, int *prev, const realT *costSum,
                                         int offsetX, int offsetY, int lenX, int lenY, int X, int Y)
{
  extern __shared__ int shared[];
  realT *minL = (realT *)(shared);
  realT *minR = (realT *)(minL + lenY);
  int *pL = (int *)(minR + lenY);
  int *pR = (int *)(pL + lenY);

  int offset = offsetY * X + (offsetX + blockIdx.x);
  for (int cur = threadIdx.x; cur < lenY; cur += blockDim.x)
  {
    minL[cur] = dist[offset + cur * X] - costSum[offset + cur * X];
    minR[lenY - 1 - cur] = dist[offset + cur * X] + costSum[offset + cur * X];
    pL[cur] = cur;
    pR[lenY - 1 - cur] = cur;
  }
  __syncthreads();
  for (int d = 0; (1 << d) < lenY; d++)
  {
    for (int cur = threadIdx.x; cur < lenY / 2; cur += blockDim.x)
    {
      int dst = (cur >> d << (d + 1) | (1 << d)) | (cur & ((1 << d) - 1));
      int src = (dst >> d << d) - 1;
      if (dst < lenY)
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
  for (int cur = threadIdx.x; cur < lenY; cur += blockDim.x)
  {
    if (minL[cur] + costSum[offset + cur * X] > minR[lenY - 1 - cur] - costSum[offset + cur * X])
    {
      dist[offset + cur * X] = minR[lenY - 1 - cur] - costSum[offset + cur * X];
      prev[offset + cur * X] = offset + pR[lenY - 1 - cur] * X;
    }
    else
    {
      dist[offset + cur * X] = minL[cur] + costSum[offset + cur * X];
      prev[offset + cur * X] = offset + pL[cur] * X;
    }
  }
}

__global__ static void cleanDist(realT *dist, const int *mark,
                                 int offsetX, int offsetY, int lenX, int lenY, int X, int Y)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lenX || y >= lenY)
    return;
  x += offsetX;
  y += offsetY;
  dist[x + y * X] = mark[x + y * X] ? 0.f : INFINITY_DISTANCE;
}

__global__ static void setRootPin(int *mark, int *isRoutedPin, const int *pinIndices, int numPins)
{
  for (int i = 0; i < numPins; i++)
    isRoutedPin[i] = 0;
  isRoutedPin[0] = 1;
  mark[pinIndices[0]] = 1;
}

__global__ static void tracePath(int *mark, int *isRoutedPin, int *routes, const realT *dist, const int *allPrev,
                                 const int *pinIndices, int numPins, int numTurns, int X, int Y)
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
  for (int t = numTurns - 1; t >= 0; t--)
  {
    int prevIdx = allPrev[t * X * Y + idx];
    if (prevIdx == idx)
      continue;
    int startIdx = min(idx, prevIdx);
    int endIdx = max(idx, prevIdx);
    routes[++routes[0]] = startIdx;
    routes[++routes[0]] = endIdx;
    if (idx / X == prevIdx / X) // horizontal
    {
      for (int tmpIdx = startIdx; tmpIdx <= endIdx; tmpIdx++)
        mark[tmpIdx] = 1;
    }
    else // vertical
    {
      for (int tmpIdx = startIdx; tmpIdx <= endIdx; tmpIdx += X)
        mark[tmpIdx] = 1;
    }
    idx = prevIdx;
  }
}

// ---------------------------------
// BasicGamer2D
// ---------------------------------

BasicGamer2D::BasicGamer2D(int X, int Y, int maxNumPins)
    : X(X), Y(Y), numPins(0), maxNumPins(maxNumPins)
{
  devCostSum = cuda_make_unique<realT[]>(2 * X * Y);
  devDist = cuda_make_unique<realT[]>(X * Y);
  devAllPrev = cuda_make_unique<int[]>(MAX_NUM_TURNS * X * Y);
  devMark = cuda_make_unique<int[]>(X * Y);

  devRoutes = cuda_make_shared<int[]>(maxNumPins * MAX_ROUTE_LEN_PER_PIN);
  devIsRoutedPin = cuda_make_unique<int[]>(maxNumPins);
  devPinIndices = cuda_make_unique<int[]>(maxNumPins);
}

bool BasicGamer2D::getIsRouted() const
{
  std::vector<int> isRoutePin(numPins);
  checkCudaErrors(cudaMemcpy(isRoutePin.data(), devIsRoutedPin.get(), numPins * sizeof(int), cudaMemcpyDeviceToHost));
  return std::reduce(isRoutePin.begin(), isRoutePin.end(), 1, [](int x, int y)
                     { return x & y; });
}

void BasicGamer2D::route(const std::vector<int> &pin2DIndices, int sweepTurns)
{
  route(pin2DIndices, sweepTurns, utils::BoxT<int>(0, 0, X, Y));
}

void BasicGamer2D::route(const std::vector<int> &pin2DIndices, int numTurns, const utils::BoxT<int> &box)
{
  int offsetX = box.lx();
  int offsetY = box.ly();
  int lenX = box.width();
  int lenY = box.height();
  int maxlen = std::max(lenX, lenY);

  numPins = static_cast<int>(pin2DIndices.size());
  checkCudaErrors(cudaMemset(devIsRoutedPin.get(), 0, numPins * sizeof(int)));
  checkCudaErrors(cudaMemcpy(devPinIndices.get(), pin2DIndices.data(), numPins * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(devRoutes.get(), 0, numPins * MAX_ROUTE_LEN_PER_PIN * sizeof(int)));
  checkCudaErrors(cudaMemset(devMark.get(), 0, X * Y * sizeof(int)));

  // 2d cost sum
  calculateWireCostSumHorizontal<<<lenY, std::min(1024, lenX / 2), lenX * sizeof(realT)>>>(
      devCostSum.get(), devCost.get(), offsetX, offsetY, lenX, lenY, X, Y);
  calculateWireCostSumVertical<<<lenX, std::min(1024, lenY / 2), lenY * sizeof(realT)>>>(
      devCostSum.get() + X * Y, devCost.get() + X * Y, offsetX, offsetY, lenX, lenY, X, Y);

  // maze routing
  setRootPin<<<1, 1>>>(devMark.get(), devIsRoutedPin.get(), devPinIndices.get(), numPins);
  cleanDist<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, 1), dim3(32, 32, 1)>>>(devDist.get(), devMark.get(), offsetX, offsetY, lenX, lenY, X, Y);
  for (int iter = 1; iter < pin2DIndices.size(); iter++)
  {
    for (int turn = 0; turn < numTurns; turn++)
    {
      if (turn & 1)
        sweepWireVertical<<<lenX, std::min(1024, lenY / 2), lenY * (2 * sizeof(realT) + 2 * sizeof(int))>>>(
            devDist.get(), devAllPrev.get() + turn * X * Y, devCostSum.get() + X * Y, offsetX, offsetY, lenX, lenY, X, Y);
      else
        sweepWireHorizontal<<<lenY, std::min(1024, lenX / 2), lenX * (2 * sizeof(realT) + 2 * sizeof(int))>>>(
            devDist.get(), devAllPrev.get() + turn * X * Y, devCostSum.get(), offsetX, offsetY, lenX, lenY, X, Y);
    }
    tracePath<<<1, 1>>>(devMark.get(), devIsRoutedPin.get(), devRoutes.get(), devDist.get(), devAllPrev.get(), devPinIndices.get(), numPins, numTurns, X, Y);
    cleanDist<<<dim3((lenX + 31) / 32, (lenY + 31) / 32, 1), dim3(32, 32, 1)>>>(devDist.get(), devMark.get(), offsetX, offsetY, lenX, lenY, X, Y);
  }
}
#endif