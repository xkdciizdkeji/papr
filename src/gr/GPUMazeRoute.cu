#include "GPUMazeRoute.h"
#include <cuda_runtime.h>
#include "../utils/helper_cuda.h"
#include "../utils/helper_math.h"
#include <numeric>

// DIRECTION的说明
// DIRECTION = 0 => 第0层顺序是X,Y（纵向）
// DIRECTION = 1 => 第0层顺序是Y,X（横向）
// (layerIdx & 1) ^ DIRECTION = 0 => 第layerIdx层顺序是X, Y（纵向）
// (layerIdx & 1) ^ DIRECTION = 1 => 第layerIdx层顺序是Y, X（横向）

// 坐标编号的说明
// gcell、wire、via的编号通过xyzToIdx、idxToXYZ函数实现
// gcell坐标(x,y,z)的范围是: 0 <= x < X, 0 <= y < Y, 0 <= z < LAYER
// wire坐标(x,y,z)表示连接(x-1,y,z)->(x,y,z)或者(x,y-1,z)->(x,y,z)的wire，却决于第z层的方向
// via坐标(x,y,z)表示连接(x,y,z-1)->(x,y,z)的via

// routes、routesOffset的说明
// routes: routes[routesOffset[netId] ... routesOffset[netId + 1]] 长度为 net.numPins * MAX_ROUTE_LEN_PER_PIN
// 记录了netId对应net的所有的连线
// 1. 其中 routes[routesOffset[netId]] 记录了该net所占routes数组的空间大小
// 2. (routes + routesOffset[netId])[1+3i,1+3i+1,1+3i+2] 记录了 (lower_pos, higher_pos)

// allpins的说明
// allpins: 大小为ALLPIN_STRIDE * MAX_BATCH_SIZE，即一个batch中每一个net都有pins数据结构，其中ALLPIN_STRIDE是所有net的最大引脚数
// 令pins = allpins + netId * ALLPIN_STRIDE，则pins记录了batch中第netId个net的引脚触点信息
// pins[0] = net的引脚数量
// pins[1] = net的第1个引脚位置
// pins[2] = net的第2个引脚位置

// treeMap的说明
// 对于idx处的gcell，如果
// 1. treeMap[idx] & TREEMAP_NODE_BIT 说明这个gcell将成为GRTree的一个结点
// 2. treeMap[idx] & TREEMAP_LEFT_BIT 说明这个gcell的左边存在一个node
// 3. treeMap[idx] & TREEMAP_RIGHT_BIT 说明这个gcell的右边存在一个node
// 以此类推

// treeDesc的说明
// treeDesc存储了一个GRTree的BFS的结果
// (TREEDESC_ROOT_NODE, idx)表示root的idx
// (TREEDESC_NODE_END, 0)表示一个树结点的结束
// (TREEDESC_FROM_LEFT, idx)表示idx的父结点在其左边
// 以此类推

// Note: 使用treeDesc的cuda kernel已被cpu版本的extractGRTree函数取代

constexpr int BLOCK_SIZE = 64;
constexpr int BLOCK_NUMBER(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }
constexpr int BLOCK_SIZE_X = 8;
constexpr int BLOCK_SIZE_Y = 8;
constexpr int BLOCK_NUMBER_X(int x) { return (x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; }
constexpr int BLOCK_NUMBER_Y(int y) { return (y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y; }
constexpr int REDUCED_SIZE = 1024;
constexpr int MAX_BATCH_SIZE = 1;          // 一个batch至多有多少个net
constexpr int MAX_ROUTE_LEN_PER_PIN = 150; // 估计每一个pin平均最多会占多少routes
constexpr int MAX_NUM_LAYER = 20;          // 估计最大的层数
constexpr float INF = 1e20f;               // 无穷大

// ------------------------------
// CUDA Device Function
// ------------------------------

__host__ __device__ int3 idxToXYZ(int idx, int DIRECTION, int N)
{
  int layer = idx / N / N;
  return (layer & 1) ^ DIRECTION ? make_int3(idx % N, (idx / N) % N, layer)
                                 : make_int3((idx / N) % N, idx % N, layer);
}

__host__ __device__ int xyzToIdx(int x, int y, int z, int DIRECTION, int N)
{
  return (z & 1) ^ DIRECTION ? z * N * N + y * N + x
                             : z * N * N + x * N + y;
}

__host__ __device__ float logistic(float x, float slope)
{
  return 1.f / (1.f + expf(x * slope));
}

// ------------------------------
// CUDA Kernel
// ------------------------------

__global__ void calculateWireCostSum(float *wireCostSum, const float *wireCost, int DIRECTION, int N, int X, int Y, int LAYER)
{
  extern __shared__ float sum[];
  for (int i = 1; i < LAYER; i++)
  {
    int offset = i * N * N + blockIdx.x * N;
    int len = (i & 1) ^ DIRECTION ? X : Y;
    for(int cur = threadIdx.x; cur < len; cur += blockDim.x)
      sum[cur] = wireCost[offset + cur];
    __syncthreads();
    for (int d = 0; (1 << d) < len; d++)
    {
      for(int cur = threadIdx.x; cur < (len + 1) / 2; cur += blockDim.x)
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
    for(int cur = threadIdx.x; cur < len; cur += blockDim.x)
      wireCostSum[offset + cur] = sum[cur];
    __syncthreads();
  }
}

__global__ void cleanDistPrev(float *dist, int *prev, int isFirstIteration, int N, int LAYER)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= LAYER * N * N)
    return;
  if (isFirstIteration || dist[idx] > 0)
  {
    dist[idx] = INF;
    prev[idx] = idx;
  }
}

__global__ void setRootPin(float *dist, int *prev, int *isRoutedPin, const int *allpins)
{
  for (int i = 0; i < allpins[0]; i++)
    isRoutedPin[i] = 0;
  isRoutedPin[0] = 1;
  dist[allpins[1]] = 0.f;
  prev[allpins[1]] = allpins[1];
}

__global__ void sweepWire(float *dist, int *prev, const float *costSum, int DIRECTION, int N, int X, int Y, int LAYER)
{
  extern __shared__ int shared[];
  float *minL = (float *)(shared);
  float *minR = (float *)(shared + N);
  int *pL = shared + 2 * N;
  int *pR = shared + 3 * N;

  for (int i = 1; i < LAYER; i++)
  {
    int offset = i * N * N + blockIdx.x * N;
    int len = (i & 1) ^ DIRECTION ? X : Y;
    for(int cur = threadIdx.x; cur < len; cur += blockDim.x)
    {
      minL[cur] = dist[offset + cur] - costSum[offset + cur];
      minR[len - 1 - cur] = dist[offset + cur] + costSum[offset + cur];
      pL[cur] = cur;
      pR[len - 1 - cur] = cur;
    }
    __syncthreads();
    for (int d = 0; (1 << d) < len; d++)
    {
      // 对于长度为N的数组，需要(N+1)/2个线程工作
      for(int cur = threadIdx.x; cur < (len + 1) / 2; cur += blockDim.x)
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
    for(int cur = threadIdx.x; cur < len; cur += blockDim.x)
    {
      if (cur < len)
      {
        float val = minL[cur] + costSum[offset + cur];
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
    }
    __syncthreads();
  }
}

__global__ void sweepVia(float *dist, int *prev, const float *viaCost, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= X || y >= Y)
    return;

  int p[MAX_NUM_LAYER];
  for (int i = 0; i < LAYER; i++)
    p[i] = i;
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

// 回溯一条路径
__global__ void tracePath(float *dist, int *prev, int *isRoutedPin, int *routes, const int *allpins, int DIRECTION, int N, int X, int Y, int LAYER)
{
  float minDist = INF;
  int pinId = -1, idx = -1;

  // 寻找距离最短的未被连线的pin
  for (int i = 0; i < allpins[0]; i++)
  {
    if (!isRoutedPin[i])
    {
      int p = allpins[i + 1];
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
  while (dist[idx] > 0)
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
        {
          dist[tmpIdx] = 0.f;
          prev[tmpIdx] = tmpIdx;
        }
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
        {
          dist[tmpIdx] = 0.f;
          prev[tmpIdx] = tmpIdx;
        }
      }
    }
    idx = prevIdx;
    x = prevX;
    y = prevY;
    z = prevZ;
  }
}

__global__ void markOverflowNet(int *isOverflowNet, const float *demand, const float *capacity, const int *routes, const int *routesOffset, int NUMNET, int DIRECTION, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NUMNET)
    return;

  routes += routesOffset[idx];
  isOverflowNet[idx] = 0;
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    int startZ = startIdx / N / N;
    int endZ = endIdx / N / N;
    if (startZ == endZ) // only check wires
    {
      for (int j = startIdx + 1; j <= endIdx; j++)
      {
        if (demand[j] > capacity[j])
        {
          isOverflowNet[idx] = 1;
          return;
        }
      }
    }
  }
}

__global__ void commitRoutes(float *demand, const float *hEdgeLengths, const float *vEdgeLengths, const float *layerMinLengths, const int *routes,
                             const int *routesOffset, const int *netIndices, float viaMultiplier, int reverse, int NUMNET, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NUMNET)
    return;

  int netId = netIndices[idx];
  routes += routesOffset[netId];
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ == endZ) // wire
    {
      for (int j = startIdx + 1; j <= endIdx; j++)
        atomicAdd(demand + j, reverse ? -1.f : 1.f);
    }
    else // vias
    {
      for (int z = startZ, x = startX, y = startY; z <= endZ; z++)
      {
        int tmpIdx = xyzToIdx(x, y, z, DIRECTION, N);
        float leftEdgeLength = ((z & 1) ^ DIRECTION) ? (x >= 1 ? hEdgeLengths[x] : 0.f) : (y >= 1 ? vEdgeLengths[y] : 0.f);
        float rightEdgeLength = ((z & 1) ^ DIRECTION) ? (x < X - 1 ? hEdgeLengths[x + 1] : 0.f) : (y < Y - 1 ? vEdgeLengths[y + 1] : 0.f);
        float vd = layerMinLengths[z] / (leftEdgeLength + rightEdgeLength) * viaMultiplier;
        vd = (z == startZ || z == endZ ? vd : 2.f * vd);
        vd = reverse ? -vd : vd;
        if (leftEdgeLength > 0.f)
          atomicAdd(demand + tmpIdx, vd);
        if (rightEdgeLength > 0.f)
          atomicAdd(demand + tmpIdx + 1, vd);
      }
    }
  }
}

// Note: cuda和cpu分别计算的wire cost之间最大误差为0.000134，via cost之间误差最大误差为0.022000
__global__ void calculateWireViaCost(float *wireCost, float *viaCost, const float *demand, const float *capacity, const float *hEdgeLengths, const float *vEdgeLengths, const float *layerMinLengths,
                                     const float *unitLengthShortCosts, float unitLengthWireCost, float unitViaCost, float logisticSlope, float viaMultiplier, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
  if (x >= X || y >= Y || z >= LAYER)
    return;

  // wire cost
  float edgeLength = ((z & 1) ^ DIRECTION) ? (x >= 1 ? hEdgeLengths[x] : 0.f) : (y >= 1 ? vEdgeLengths[y] : 0.f);
  float logisticFactor = capacity[idx] < 1.f ? 1.f : logistic(capacity[idx] - demand[idx], logisticSlope);
  wireCost[idx] = edgeLength * (unitLengthWireCost + unitLengthShortCosts[z] * logisticFactor);

  // via cost
  float vc = 0.f;
  if (z >= 1)
  {
    vc += unitViaCost;
#pragma unroll
    for (int l = z - 1; l <= z; l++)
    {
      int leftIdx = xyzToIdx(x, y, l, DIRECTION, N);
      int rightIdx = leftIdx + 1;
      float leftEdgeLength = ((l & 1) ^ DIRECTION) ? (x >= 1 ? hEdgeLengths[x] : 0.f) : (y >= 1 ? vEdgeLengths[y] : 0.f);
      float rightEdgeLength = ((l & 1) ^ DIRECTION) ? (x < X - 1 ? hEdgeLengths[x + 1] : 0.f) : (y < Y - 1 ? vEdgeLengths[y + 1] : 0.f);
      float vd = layerMinLengths[l] / (leftEdgeLength + rightEdgeLength) * viaMultiplier;
      if (leftEdgeLength > 0.f)
      {
        float leftLogisticFactor = capacity[leftIdx] < 1.f ? 1.f : logistic(capacity[leftIdx] - demand[leftIdx], logisticSlope);
        vc += vd * leftEdgeLength * (unitLengthWireCost + unitLengthShortCosts[l] * leftLogisticFactor);
      }
      if (rightEdgeLength > 0.f)
      {
        float rightLogisticFactor = capacity[rightIdx] < 1.f ? 1.f : logistic(capacity[rightIdx] - demand[rightIdx], logisticSlope);
        vc += vd * rightEdgeLength * (unitLengthWireCost + unitLengthShortCosts[l] * rightLogisticFactor);
      }
    }
  }
  viaCost[idx] = vc;
}

template<class T>
__global__ void checkIsIdentical(int *out, const T *in1, const T *in2, int n)
{
  extern __shared__ int share[];

  int same = 1;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    same &= (in1[i] == in2[i]);
  
  share[threadIdx.x] = same;
  __syncthreads();

  for (int i = blockDim.x / 2; i > 32; i >>= 1)
  {
    if (threadIdx.x < i)
      share[threadIdx.x] &= share[threadIdx.x + i];
    __syncthreads();
  }

  if(threadIdx.x < 32)
  {
    share[threadIdx.x] &= share[threadIdx.x + 32];
    share[threadIdx.x] &= share[threadIdx.x + 16];
    share[threadIdx.x] &= share[threadIdx.x + 8];
    share[threadIdx.x] &= share[threadIdx.x + 4];
    share[threadIdx.x] &= share[threadIdx.x + 2];
    share[threadIdx.x] &= share[threadIdx.x + 1];
  }
  if (threadIdx.x == 0)
    out[blockIdx.x] = share[0];
}

// --------------------------------
// GPUMazeRoute
// --------------------------------

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
    : nets(nets), gridGraph(graph), parameters(params)
{
  DIRECTION = gridGraph.getLayerDirection(0) == MetalLayer::H;
  LAYER = gridGraph.getNumLayers();
  X = gridGraph.getSize(0);
  Y = gridGraph.getSize(1);
  N = std::max(X, Y);
  NUMNET = nets.size();
  ALLPIN_STRIDE = 1 + std::max_element(nets.begin(), nets.end(), [](const GRNet &net1, const GRNet &net2)
                                       { return net1.getNumPins() < net2.getNumPins(); })->getNumPins();

  unitLengthWireCost = gridGraph.getUnitLengthWireCost();
  unitViaCost = gridGraph.getUnitViaCost();
  logisticSlope = parameters.cost_logistic_slope;
  viaMultiplier = parameters.via_multiplier;

  // 初始化hEdgeLengths, vEdgeLengths内存显存
  auto hEdgeLengths = std::make_unique<float[]>(X);
  auto vEdgeLengths = std::make_unique<float[]>(Y);
  hEdgeLengths[0] = vEdgeLengths[0] = 0.f;
  for (int x = 1; x < X; x++)
    hEdgeLengths[x] = gridGraph.getEdgeLength(MetalLayer::H, x - 1);
  for (int y = 1; y < Y; y++)
    vEdgeLengths[y] = gridGraph.getEdgeLength(MetalLayer::V, y - 1);
  checkCudaErrors(cudaMalloc(&devHEdgeLengths, X * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devVEdgeLengths, Y * sizeof(float)));
  checkCudaErrors(cudaMemcpy(devHEdgeLengths, hEdgeLengths.get(), X * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devVEdgeLengths, vEdgeLengths.get(), Y * sizeof(float), cudaMemcpyHostToDevice));

  // 初始化layerMinLengths, unitLengthShortCosts内存显存
  auto layerMinLengths = std::make_unique<float[]>(LAYER);
  auto unitLengthShortCosts = std::make_unique<float[]>(LAYER);
  for (int l = 0; l < LAYER; l++)
  {
    layerMinLengths[l] = gridGraph.getLayerMinLength(l);
    unitLengthShortCosts[l] = gridGraph.getUnitLengthShortCost(l);
  }
  checkCudaErrors(cudaMalloc(&devLayerMinLengths, LAYER * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devUnitLengthShortCosts, LAYER * sizeof(float)));
  checkCudaErrors(cudaMemcpy(devLayerMinLengths, layerMinLengths.get(), LAYER * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devUnitLengthShortCosts, unitLengthShortCosts.get(), LAYER * sizeof(float), cudaMemcpyHostToDevice));

  // 初始化capacity, demand内存显存
  auto capacity = std::make_unique<float[]>(LAYER * N * N);
  auto demand = std::make_unique<float[]>(LAYER * N * N);
  for (int z = 0; z < LAYER; z++)
  {
    if ((z & 1) ^ DIRECTION)
    {
      for (int y = 0; y < Y; y++)
        for (int x = 1; x < X; x++)
        {
          capacity[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x - 1, y).capacity;
          demand[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x - 1, y).demand;
        }
    }
    else
    {
      for (int x = 0; x < X; x++)
        for (int y = 1; y < Y; y++)
        {
          capacity[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x, y - 1).capacity;
          demand[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x, y - 1).demand;
        }
    }
  }
  checkCudaErrors(cudaMalloc(&devCapacity, LAYER * N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devDemand, LAYER * N * N * sizeof(float)));
  checkCudaErrors(cudaMemcpy(devCapacity, capacity.get(), LAYER * N * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devDemand, demand.get(), LAYER * N * N * sizeof(float), cudaMemcpyHostToDevice));

  // 初始化rootIndices, isRoutedNet, routes, routesOffset显存内存
  cpuRootIndices = (int *)malloc(NUMNET * sizeof(int));
  cpuIsRoutedNet = (int *)malloc(NUMNET * sizeof(int));
  cpuRoutesOffset = (int *)malloc((NUMNET + 1) * sizeof(int));
  cpuRoutesOffset[0] = 0;
  for (int i = 0; i < NUMNET; i++)
    cpuRoutesOffset[i + 1] = cpuRoutesOffset[i] + nets[i].getNumPins() * MAX_ROUTE_LEN_PER_PIN;
  cpuRoutes = (int *)malloc(cpuRoutesOffset[NUMNET] * sizeof(int));
  // 根据GRNets设置routes
  for (int i = 0; i < NUMNET; i++)
  {
    const auto &net = nets[i];
    int *routes = cpuRoutes + cpuRoutesOffset[i];
    auto tree = net.getRoutingTree();
    if (tree == nullptr)
    {
      cpuRootIndices[i] = 0;
      cpuIsRoutedNet[i] = 0;
      routes[0] = 0;
    }
    else
    {
      cpuRootIndices[i] = xyzToIdx(tree->x, tree->y, tree->layerIdx, DIRECTION, N);
      cpuIsRoutedNet[i] = 1;
      routes[0] = 0;
      GRTreeNode::preorder(tree, [&](std::shared_ptr<GRTreeNode> node)
      {
        int nodeIdx = xyzToIdx(node->x, node->y, node->layerIdx, DIRECTION, N);
        for(auto child : node->children)
        {
          int childIdx = xyzToIdx(child->x, child->y, child->layerIdx, DIRECTION, N);
          routes[++routes[0]] = std::min(nodeIdx, childIdx);
          routes[++routes[0]] = std::max(nodeIdx, childIdx);
        }
      });
      if (routes[0] > net.getNumPins() * MAX_ROUTE_LEN_PER_PIN)
      {
        ::utils::log() << "ERROR: Not enough routes size for net(id=" << net.getIndex() << "), "
                       << "where its routes size per pin = " << (routes[0] + net.getNumPins() - 1) / net.getNumPins()
                       << " but MAX_ROUTE_LEN_PER_PIN = " << MAX_ROUTE_LEN_PER_PIN << "\n";
        exit(-1);
      }
    }
  }
  checkCudaErrors(cudaMalloc(&devRoutes, cpuRoutesOffset[NUMNET] * sizeof(int)));
  checkCudaErrors(cudaMalloc(&devRoutesOffset, (NUMNET + 1) * sizeof(int)));
  checkCudaErrors(cudaMemcpy(devRoutes, cpuRoutes, cpuRoutesOffset[NUMNET] * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devRoutesOffset, cpuRoutesOffset, (NUMNET + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // 初始化isOverflowNet, netIndices内存显存
  cpuIsOverflowNet = (int *)malloc(NUMNET * sizeof(int));
  cpuNetIndices = (int *)malloc(NUMNET * sizeof(int));
  checkCudaErrors(cudaMalloc(&devIsOverflowNet, NUMNET * sizeof(int)));
  checkCudaErrors(cudaMalloc(&devNetIndices, NUMNET * sizeof(int)));

  // 初始化allpins, isRoutedPin内存显存
  cpuAllpins = (int *)malloc(MAX_BATCH_SIZE * ALLPIN_STRIDE * sizeof(int));
  cpuIsRoutedPin = (int *)malloc(MAX_BATCH_SIZE * ALLPIN_STRIDE * sizeof(int));
  checkCudaErrors(cudaMalloc(&devAllpins, MAX_BATCH_SIZE * ALLPIN_STRIDE * sizeof(int)));
  checkCudaErrors(cudaMalloc(&devIsRoutedPin, MAX_BATCH_SIZE * ALLPIN_STRIDE * sizeof(int)));

  // 初始化wireCost, wireCostSum, viaCost显存
  checkCudaErrors(cudaMalloc(&devWireCost, LAYER * N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devWireCostSum, LAYER * N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devViaCost, LAYER * N * N * sizeof(float)));

  // 初始化dist, prev显存
  checkCudaErrors(cudaMalloc(&devDist, MAX_BATCH_SIZE * LAYER * N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devPrev, MAX_BATCH_SIZE * LAYER * N * N * sizeof(int)));

  // 初始化isIdentical, lastPrev
  cpuIsIdentical = (int *)malloc(REDUCED_SIZE * sizeof(int));
  checkCudaErrors(cudaMalloc(&devIsIdentical, REDUCED_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc(&devLastPrev, MAX_BATCH_SIZE * LAYER * N * N * sizeof(int)));

  // 标记overflow net
  markOverflowNet<<<BLOCK_NUMBER(NUMNET), BLOCK_SIZE>>>(devIsOverflowNet, devDemand, devCapacity, devRoutes, devRoutesOffset, NUMNET, DIRECTION, N);
  checkCudaErrors(cudaMemcpy(cpuIsOverflowNet, devIsOverflowNet, NUMNET * sizeof(int), cudaMemcpyDeviceToHost));
}

GPUMazeRoute::~GPUMazeRoute()
{
  checkCudaErrors(cudaFree(devHEdgeLengths));
  checkCudaErrors(cudaFree(devVEdgeLengths));
  checkCudaErrors(cudaFree(devLayerMinLengths));
  checkCudaErrors(cudaFree(devUnitLengthShortCosts));

  checkCudaErrors(cudaFree(devCapacity));
  checkCudaErrors(cudaFree(devDemand));

  free(cpuRootIndices);
  free(cpuIsRoutedNet);

  free(cpuRoutes);
  checkCudaErrors(cudaFree(devRoutes));
  free(cpuRoutesOffset);
  checkCudaErrors(cudaFree(devRoutesOffset));

  free(cpuIsOverflowNet);
  checkCudaErrors(cudaFree(devIsOverflowNet));
  free(cpuNetIndices);
  checkCudaErrors(cudaFree(devNetIndices));

  free(cpuAllpins);
  checkCudaErrors(cudaFree(devAllpins));
  free(cpuIsRoutedPin);
  checkCudaErrors(cudaFree(devIsRoutedPin));

  checkCudaErrors(cudaFree(devWireCost));
  checkCudaErrors(cudaFree(devWireCostSum));
  checkCudaErrors(cudaFree(devViaCost));

  checkCudaErrors(cudaFree(devDist));
  checkCudaErrors(cudaFree(devPrev));

  free(cpuIsIdentical);
  checkCudaErrors(cudaFree(devIsIdentical));
  checkCudaErrors(cudaFree(devLastPrev));
}

void GPUMazeRoute::route(const std::vector<int> &netIndices, int sweepTurns)
{
  // 撤销
  int numNetToRoute = netIndices.size();
  checkCudaErrors(cudaMemcpy(devNetIndices, netIndices.data(), numNetToRoute * sizeof(int), cudaMemcpyHostToDevice));
  commitRoutes<<<BLOCK_NUMBER(numNetToRoute), BLOCK_SIZE>>>(devDemand, devHEdgeLengths, devVEdgeLengths, devLayerMinLengths, devRoutes,
                                                            devRoutesOffset, devNetIndices, viaMultiplier, 1, numNetToRoute, DIRECTION, N, X, Y, LAYER);
  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < netIndices.size(); i++)
  {
    int netId = netIndices[i];
    cpuIsRoutedNet[netId] = 0;

    // 初始化allpins
    std::vector<int> points = selectAccessPoints(nets[netId]);
    cpuAllpins[0] = 0;
    for (int p : points)
      cpuAllpins[++cpuAllpins[0]] = p;
    checkCudaErrors(cudaMemcpy(devAllpins, cpuAllpins, ALLPIN_STRIDE * sizeof(int), cudaMemcpyHostToDevice));
    // 初始化isRoutedPin
    checkCudaErrors(cudaMemset(devIsRoutedPin, 0, ALLPIN_STRIDE * sizeof(int)));
    // 初始化routes
    int *devNetRoutes = devRoutes + cpuRoutesOffset[netId];
    int *cpuNetRoutes = cpuRoutes + cpuRoutesOffset[netId];
    int netRoutesLen = cpuRoutesOffset[netId + 1] - cpuRoutesOffset[netId];
    checkCudaErrors(cudaMemset(devNetRoutes, 0, netRoutesLen * sizeof(int)));

    ::utils::log() << "gamer routing. netId = " << netId << ". #pins = " << points.size() << ". #turns = " << (points.size() - 1) * sweepTurns << "\n";

    // compute wireCost, viaCost
    calculateWireViaCost<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>>(
        devWireCost, devViaCost, devDemand, devCapacity, devHEdgeLengths, devVEdgeLengths, devLayerMinLengths,
        devUnitLengthShortCosts, unitLengthWireCost, unitViaCost, logisticSlope, viaMultiplier, DIRECTION, N, X, Y, LAYER);
    calculateWireCostSum<<<N, (N + 1) / 2, N * sizeof(float), 0>>>(devWireCostSum, devWireCost, DIRECTION, N, X, Y, LAYER);

    // maze routing
    cleanDistPrev<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>>(devDist, devPrev, 1, N, LAYER);
    setRootPin<<<1, 1>>>(devDist, devPrev, devIsRoutedPin, devAllpins);
    cpuRootIndices[netId] = cpuAllpins[1];
    for (int iter = 1; iter < points.size(); iter++)
    {
      for (int turn = 0; turn < sweepTurns; turn++)
      {
        sweepVia<<<dim3(BLOCK_NUMBER_X(X), BLOCK_NUMBER_Y(Y)), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(devDist, devPrev, devViaCost, DIRECTION, N, X, Y, LAYER);
        sweepWire<<<N, (N + 1) / 2, 4 * N * sizeof(float), 0>>>(devDist, devPrev, devWireCostSum, DIRECTION, N, X, Y, LAYER);
      }
      // for (int turn = 0; turn < 2; turn++)
      // {
      //   sweepVia<<<dim3(BLOCK_NUMBER_X(X), BLOCK_NUMBER_Y(Y)), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(devDist, devPrev, devViaCost, DIRECTION, N, X, Y, LAYER);
      //   sweepWire<<<N, (N + 1) / 2, 4 * N * sizeof(float), 0>>>(devDist, devPrev, devWireCostSum, DIRECTION, N, X, Y, LAYER);
      // }
      // for(int turn = 2; turn < sweepTurns; turn++)
      // {
      //   checkCudaErrors(cudaMemcpy(devLastPrev, devPrev, LAYER * N * N * sizeof(int), cudaMemcpyDeviceToDevice));
      //   sweepVia<<<dim3(BLOCK_NUMBER_X(X), BLOCK_NUMBER_Y(Y)), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(devDist, devPrev, devViaCost, DIRECTION, N, X, Y, LAYER);
      //   sweepWire<<<N, (N + 1) / 2, 4 * N * sizeof(float), 0>>>(devDist, devPrev, devWireCostSum, DIRECTION, N, X, Y, LAYER);
      //   checkIsIdentical<<<REDUCED_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int), 0>>>(devIsIdentical, devLastPrev, devPrev, LAYER * N * N);
      //   checkCudaErrors(cudaMemcpy(cpuIsIdentical, devIsIdentical, REDUCED_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
      //   if(std::reduce(cpuIsIdentical, cpuIsIdentical + REDUCED_SIZE, 1, [](int x, int y) { return x & y; }))
      //     break;
      // }
      sweepVia<<<dim3(BLOCK_NUMBER_X(X), BLOCK_NUMBER_Y(Y)), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(devDist, devPrev, devViaCost, DIRECTION, N, X, Y, LAYER);
      tracePath<<<1, 1>>>(devDist, devPrev, devIsRoutedPin, devNetRoutes, devAllpins, DIRECTION, N, X, Y, LAYER);
      cleanDistPrev<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>>(devDist, devPrev, 0, N, LAYER);
    }

    // commit route
    commitRoutes<<<1, 1>>>(devDemand, devHEdgeLengths, devVEdgeLengths, devLayerMinLengths, devRoutes,
                           devRoutesOffset, devNetIndices + i, viaMultiplier, 0, 1, DIRECTION, N, X, Y, LAYER);

    // copy routes to cpu
    checkCudaErrors(cudaMemcpy(cpuNetRoutes, devNetRoutes, netRoutesLen * sizeof(int), cudaMemcpyDeviceToHost));
    // check every pin is routed
    checkCudaErrors(cudaMemcpy(cpuIsRoutedPin, devIsRoutedPin, ALLPIN_STRIDE * sizeof(int), cudaMemcpyDeviceToHost));
    for (int pinId = 0; pinId < cpuAllpins[0]; pinId++) // 只要sweepTurn >= 2就不会出现unrouted pin
    {
      if (!cpuIsRoutedPin[pinId])
      {
        ::utils::log() << "ERROR: Not all pins of net(id=" << netId << ") are routed\n";
        exit(-1);
      }
    }
    cpuIsRoutedNet[netId] = 1;
  }

  // 标记overflow net
  markOverflowNet<<<BLOCK_NUMBER(NUMNET), BLOCK_SIZE>>>(devIsOverflowNet, devDemand, devCapacity, devRoutes, devRoutesOffset, NUMNET, DIRECTION, N);
  checkCudaErrors(cudaMemcpy(cpuIsOverflowNet, devIsOverflowNet, NUMNET * sizeof(int), cudaMemcpyDeviceToHost));
}

void GPUMazeRoute::commit(const std::vector<int> &netIndices)
{
  // TODO: bactching and multi-threading

  auto treeMap = std::make_unique<int[]>(LAYER * N * N);
  for (int netId : netIndices)
  {
    if(cpuIsRoutedNet[netId])
    {
      // ripup old tree
      auto oldTree = nets[netId].getRoutingTree();
      if (oldTree != nullptr)
        gridGraph.commitTree(oldTree, true);
      // commit new tree
      memset(treeMap.get(), 0, LAYER * N * N * sizeof(int));
      auto newTree = extractGRTree(treeMap.get(), cpuRoutes + cpuRoutesOffset[netId], cpuRootIndices[netId]);
      nets[netId].setRoutingTree(newTree);
      gridGraph.commitTree(newTree, false);
    }
    else
    {
      ::utils::log() << "warning: net(id=" << netId << ") is not routed\n";
    }
  }
}

void GPUMazeRoute::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  netIndices.clear();
  for (int netId = 0; netId < NUMNET; netId++)
  {
    if (cpuIsOverflowNet[netId])
      netIndices.push_back(netId);
  }
}

std::pair<std::vector<int>, std::vector<int>> GPUMazeRoute::batching(const std::vector<int> &netIndices) const
{
  ::utils::log() << "Begin Batching ...\n";
  std::vector<int> batchSizes, netIndicesToRoute;
  netIndicesToRoute.reserve(netIndices.size());

  // 按照boundingBox大小排序
  std::vector<int> s = netIndices;
  std::sort(s.begin(), s.end(), [&](int l, int r)
            { return nets[l].getBoundingBox().area() > nets[r].getBoundingBox().area(); });

  // 用boudingBox. TODO: quad-tree
  std::unordered_map<int, ::utils::BoxT<int>> batchBoxes;
  batchBoxes.reserve(MAX_BATCH_SIZE);
  constexpr int margin = 0;
  auto noConflict = [&](int netId)
  {
    auto b1 = nets[netId].getBoundingBox();
    b1.Set(std::max(0, b1.lx() - margin), std::max(0, b1.ly() - margin), 
           std::min(X, b1.hx() + margin), std::min(Y, b1.hy() + margin));
    for(const auto &b2 : batchBoxes)
    {
      if(b1.HasIntersectWith(b2.second))
        return false;
    }
    return true;
  };
  auto insert = [&](int netId)
  {
    batchBoxes.emplace(netId, nets[netId].getBoundingBox());
  };
  auto remove = [&](int netId)
  {
    batchBoxes.erase(netId);
  };

  // // 在平面上划分网格
  // std::vector<bool> vis(X * Y, false);
  // constexpr int margin = 0;
  // auto noConflict = [&](int netId)
  // {
  //   const auto &box = nets[netId].getBoundingBox();
  //   for (int i = std::max(0, box.lx() - margin), ie = std::min(X, box.hx() + margin); i < ie; i++)
  //     for (int j = std::max(0, box.ly() - margin), je = std::min(Y, box.hy() + margin); j < je; j++)
  //       if (vis[i * Y + j])
  //         return false;
  //   return true;
  // };
  // auto insert = [&](int netId)
  // {
  //   const auto &box = nets[netId].getBoundingBox();
  //   for (int i = std::max(0, box.lx() - margin), ie = std::min(X, box.hx() + margin); i < ie; i++)
  //     for (int j = std::max(0, box.ly() - margin), je = std::min(Y, box.hy() + margin); j < je; j++)
  //       vis[i * Y + j] = true;
  // };
  // auto remove = [&](int netId)
  // {
  //   const auto &box = nets[netId].getBoundingBox();
  //   for (int i = std::max(0, box.lx() - margin), ie = std::min(X, box.hx() + margin); i < ie; i++)
  //     for (int j = std::max(0, box.ly() - margin), je = std::min(Y, box.hy() + margin); j < je; j++)
  //       vis[i * Y + j] = false;
  // };

  // batching
  int lastUnroute = 0;
  while (netIndicesToRoute.size() < s.size())
  {
    int sz = netIndicesToRoute.size();
    int cnt = 0; // 连续的不能打包成batch的net个数
    for (int i = lastUnroute; i < s.size(); i++)
    {
      if (s[i] != -1)
      {
        if (noConflict(s[i]))
        {
          netIndicesToRoute.emplace_back(s[i]);
          insert(s[i]);
          s[i] = -1;
          cnt = 0;
        }
        else
          cnt++;

        if (cnt > 100)
          break;

        if (netIndicesToRoute.size() - sz == MAX_BATCH_SIZE)
          break;
      }
    }
    while (lastUnroute < s.size() && s[lastUnroute] == -1)
      lastUnroute++;
    for (int i = sz; i < netIndicesToRoute.size(); i++)
      remove(netIndicesToRoute[i]);
    batchSizes.emplace_back(static_cast<int>(netIndicesToRoute.size()) - sz);
  }
  std::reverse(batchSizes.begin(), batchSizes.end());
  std::reverse(netIndicesToRoute.begin(), netIndicesToRoute.end());

  ::utils::log() << "End Batching.\n";

  return std::make_pair(std::move(batchSizes), std::move(netIndicesToRoute));
}

std::shared_ptr<GRTreeNode> GPUMazeRoute::extractGRTree(int *treeMap, const int *routes, int rootIdx) const
{
  // 假设treeMap全设为0

  // TreeMap用到的常量
  constexpr int TREEMAP_NODE_BIT = 1;
  constexpr int TREEMAP_LEFT_BIT = 1 << 1;
  constexpr int TREEMAP_RIGHT_BIT = 1 << 2;
  constexpr int TREEMAP_DOWN_BIT = 1 << 3;
  constexpr int TREEMAP_UP_BIT = 1 << 4;

  // TreeDesc用到的常量
  constexpr int TREEDESC_ROOT_NODE = 1;
  constexpr int TREEDESC_FROM_LEFT = 2;
  constexpr int TREEDESC_FROM_RIGHT = 3;
  constexpr int TREEDESC_FROM_UP = 4;
  constexpr int TREEDESC_FROM_DOWN = 5;

  // 在TreeMap上标记
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[i + 1];
    int endIdx = routes[i + 2];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ == endZ) // wire
    {
      // 标记中间点
      for (int j = startIdx + 1; j <= endIdx - 1; j++)
        treeMap[j] |= (TREEMAP_LEFT_BIT | TREEMAP_RIGHT_BIT);
      // 标记起点
      treeMap[startIdx] |= (TREEMAP_NODE_BIT | TREEMAP_RIGHT_BIT);
      // 标记终点
      treeMap[endIdx] |= (TREEMAP_NODE_BIT | TREEMAP_LEFT_BIT);
    }
    else // via
    {
      // 标记中间点
      for (int l = startZ + 1, x = startX, y = startY; l <= endZ - 1; l++)
        treeMap[xyzToIdx(x, y, l, DIRECTION, N)] |= (TREEMAP_DOWN_BIT | TREEMAP_UP_BIT);
      // 标记起点
      treeMap[startIdx] |= (TREEMAP_NODE_BIT | TREEMAP_UP_BIT);
      // 标记终点
      treeMap[endIdx] |= (TREEMAP_NODE_BIT | TREEMAP_DOWN_BIT);
    }
  }

  // 根据treeMap构造GRTree
  std::vector<std::shared_ptr<GRTreeNode>> allnodes;
  std::vector<int> allnodesIdx;
  std::vector<int> allnodesFrom;
  auto addNewNode = [&](int newNodeIdx, int newNodeFrom, std::shared_ptr<GRTreeNode> parent)
  {
    auto [x, y, z] = idxToXYZ(newNodeIdx, DIRECTION, N);
    auto newNode = std::make_shared<GRTreeNode>(z, x, y);
    allnodes.push_back(newNode);
    allnodesIdx.push_back(newNodeIdx);
    allnodesFrom.push_back(newNodeFrom);
    if (parent != nullptr)
      parent->children.push_back(newNode);
  };
  // root
  addNewNode(rootIdx, TREEDESC_ROOT_NODE, nullptr);
  // bfs
  for (int i = 0; i < allnodes.size(); i++)
  {
    auto currNode = allnodes[i];
    int currIdx = allnodesIdx[i];
    int currFrom = allnodesFrom[i];
    // LEFT
    if ((treeMap[currIdx] & TREEMAP_LEFT_BIT) && (currFrom != TREEDESC_FROM_LEFT))
    {
      int leftIdx = currIdx - 1;
      while (!(treeMap[leftIdx] & TREEMAP_NODE_BIT))
        leftIdx--;
      addNewNode(leftIdx, TREEDESC_FROM_RIGHT, currNode);
    }
    // RIGHT
    if ((treeMap[currIdx] & TREEMAP_RIGHT_BIT) && (currFrom != TREEDESC_FROM_RIGHT))
    {
      int rightIdx = currIdx + 1;
      while (!(treeMap[rightIdx] & TREEMAP_NODE_BIT))
        rightIdx++;
      addNewNode(rightIdx, TREEDESC_FROM_LEFT, currNode);
    }
    // UP
    if ((treeMap[currIdx] & TREEMAP_UP_BIT) && (currFrom != TREEDESC_FROM_UP))
    {
      int z = currNode->layerIdx;
      int upIdx = xyzToIdx(currNode->x, currNode->y, ++z, DIRECTION, N);
      while (!(treeMap[upIdx] & TREEMAP_NODE_BIT))
        upIdx = xyzToIdx(currNode->x, currNode->y, ++z, DIRECTION, N);
      addNewNode(upIdx, TREEDESC_FROM_DOWN, currNode);
    }
    // DOWN
    if ((treeMap[currIdx] & TREEMAP_DOWN_BIT) && (currFrom != TREEDESC_FROM_DOWN))
    {
      int z = currNode->layerIdx;
      int downIdx = xyzToIdx(currNode->x, currNode->y, --z, DIRECTION, N);
      while (!(treeMap[downIdx] & TREEMAP_NODE_BIT))
        downIdx = xyzToIdx(currNode->x, currNode->y, --z, DIRECTION, N);
      addNewNode(downIdx, TREEDESC_FROM_UP, currNode);
    }
  }
  return allnodes[0];
}

std::vector<int> GPUMazeRoute::selectAccessPoints(const GRNet &net) const
{
  std::set<int> selectedAccessPoints;
  const auto &boundingBox = net.getBoundingBox();
  ::utils::PointT<int> netCenter(boundingBox.cx(), boundingBox.cy());
  for (const auto &points : net.getPinAccessPoints())
  {
    std::pair<int, int> bestAccessDist = {0, std::numeric_limits<int>::max()};
    int bestIndex = -1;
    for (int index = 0; index < points.size(); index++)
    {
      const auto &point = points[index];
      int accessibility = 0;
      if (point.layerIdx >= parameters.min_routing_layer)
      {
        unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
        accessibility += gridGraph.getEdge(point.layerIdx, point.x, point.y).capacity >= 1;
        if (point[direction] > 0)
        {
          auto lower = point;
          lower[direction] -= 1;
          accessibility += gridGraph.getEdge(lower.layerIdx, lower.x, lower.y).capacity >= 1;
        }
      }
      else
      {
        accessibility = 1;
      }
      int distance = std::abs(netCenter.x - point.x) + std::abs(netCenter.y - point.y);
      if (accessibility > bestAccessDist.first || (accessibility == bestAccessDist.first && distance < bestAccessDist.second))
      {
        bestIndex = index;
        bestAccessDist = {accessibility, distance};
      }
    }
    if (bestAccessDist.first == 0)
      ::utils::log() << "Warning: the pin is hard to access." << std::endl;
    selectedAccessPoints.insert(xyzToIdx(points[bestIndex].x, points[bestIndex].y, points[bestIndex].layerIdx, DIRECTION, N));
  }
  return std::vector<int>(selectedAccessPoints.begin(), selectedAccessPoints.end());
}
