#ifdef ENABLE_CUDA

#include "GPUMazeRouter.cuh"
#include <numeric>

// ----------------------------
// Cuda kernel
// ----------------------------

__global__ static void commitRoutes(realT *demand, const int *allRoutes, const int *allRoutesOffset, const int *netIndices,
                                    int numNets, int reverse, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > numNets)
    return;
  const int *routes = allRoutes + allRoutesOffset[netIndices[tid]];
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ == endZ) // wire
    {
      for (int idx = startIdx + 1; idx <= endIdx; idx++)
      {
#if __CUDA_ARCH__ >= 600
        atomicAdd(demand + idx, reverse ? static_cast<realT>(-1) : static_cast<realT>(1));
#else
        myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
#endif
      }
    }
    else // vias
    {
      for (int z = startZ, x = startX, y = startY; z <= endZ; z++)
      {
        int idx = xyzToIdx(x, y, z, DIRECTION, N);
        realT vd = (z == startZ || z == endZ) ? 1.f : 2.f;
#if __CUDA_ARCH__ >= 600
        if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
          atomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-vd) : static_cast<realT>(vd));
        else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
          atomicAdd(demand + idx, reverse ? static_cast<realT>(-vd) : static_cast<realT>(vd));
        else
        {
          atomicAdd(demand + idx, reverse ? static_cast<realT>(-0.5f * vd) : static_cast<realT>(0.5f * vd));
          atomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-0.5f * vd) : static_cast<realT>(0.5f * vd));
        }
#else
        if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
          myAtomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-vd) : static_cast<realT>(vd));
        else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
          myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-vd) : static_cast<realT>(vd));
        else
        {
          myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-0.5f * vd) : static_cast<realT>(0.5f * vd));
          myAtomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-0.5f * vd) : static_cast<realT>(0.5f * vd));
        }
#endif
      }
    }
  }
}

__global__ static void markOverflowNet(int *isOverflowNet, const realT *demand, const realT *capacity,
                                       const int *allRoutes, const int *allRoutesOffset, int numNets, int DIRECTION, int N)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= numNets)
    return;

  const int *routes = allRoutes + allRoutesOffset[nid];
  isOverflowNet[nid] = 0;
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    int startZ = startIdx / N / N;
    int endZ = endIdx / N / N;
    if (startZ == endZ) // only check wires
    {
      for (int idx = startIdx + 1; idx <= endIdx; idx++)
      {
        if (demand[idx] > capacity[idx])
        {
          isOverflowNet[nid] = 1;
          return;
        }
      }
    }
  }
}

__global__ static void calculateWireViaCost(realT *wireCost, realT *viaCost, const realT *demand, const realT *capacity,
                                            const realT *hEdgeLengths, const realT *vEdgeLengths, const realT *layerMinLengths, const realT *unitLengthShortCosts,
                                            realT unitLengthWireCost, realT unitViaCost, realT logisticSlope, realT viaMultiplier,
                                            int offsetX, int offsetY, int lenX, int lenY, int DIRECTION, int N, int X, int Y, int LAYER)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= lenX || y >= lenY || z >= LAYER)
    return;
  x += offsetX;
  y += offsetY;
  int idx = xyzToIdx(x, y, z, DIRECTION, N);

  // wire cost
  realT edgeLength = ((z & 1) ^ DIRECTION) ? (x > 0 ? hEdgeLengths[x] : 0.f) : (y > 0 ? vEdgeLengths[y] : 0.f);
  realT logisticFactor = capacity[idx] < 1.f ? 1.f : logistic(capacity[idx] - demand[idx], logisticSlope);
  wireCost[idx] = edgeLength * (unitLengthWireCost + unitLengthShortCosts[z] * logisticFactor);

  // via cost
  realT vc = 0.f;
  if (z >= 1)
  {
    vc += unitViaCost;
#pragma unroll
    for (int l = z - 1; l <= z; l++)
    {
      int leftIdx = xyzToIdx(x, y, l, DIRECTION, N);
      int rightIdx = leftIdx + 1;
      realT leftEdgeLength = ((l & 1) ^ DIRECTION) ? (x > 0 ? hEdgeLengths[x] : 0.f) : (y > 0 ? vEdgeLengths[y] : 0.f);
      realT rightEdgeLength = ((l & 1) ^ DIRECTION) ? (x < X - 1 ? hEdgeLengths[x + 1] : 0.f) : (y < Y - 1 ? vEdgeLengths[y + 1] : 0.f);
      realT vd = layerMinLengths[l] / (leftEdgeLength + rightEdgeLength) * viaMultiplier;
      if (leftEdgeLength > 0.f)
      {
        realT leftLogisticFactor = capacity[leftIdx] < 1.f ? 1.f : logistic(capacity[leftIdx] - demand[leftIdx], logisticSlope);
        vc += vd * leftEdgeLength * (unitLengthWireCost + unitLengthShortCosts[l] * leftLogisticFactor);
      }
      if (rightEdgeLength > 0.f)
      {
        realT rightLogisticFactor = capacity[rightIdx] < 1.f ? 1.f : logistic(capacity[rightIdx] - demand[rightIdx], logisticSlope);
        vc += vd * rightEdgeLength * (unitLengthWireCost + unitLengthShortCosts[l] * rightLogisticFactor);
      }
    }
  }
  viaCost[idx] = vc;
}

// ----------------------------------
// GPU Maze Router
// ----------------------------------

GPUMazeRouter::GPUMazeRouter(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
    : nets(nets), gridGraph(graph), parameters(params)
{
  DIRECTION = gridGraph.getLayerDirection(0) == MetalLayer::H;
  LAYER = gridGraph.getNumLayers();
  X = gridGraph.getSize(0);
  Y = gridGraph.getSize(1);
  N = std::max(X, Y);
  numNets = nets.size();
  maxNumPins = std::transform_reduce(
      nets.begin(), nets.end(), 0, [](int x, int y)
      { return std::max(x, y); },
      [](const GRNet &net)
      { return net.getNumPins(); });
  scaleX = (X + 511) / 512;
  scaleY = (Y + 511) / 512;
  // scaleX = 1;
  // scaleY = 1;

  unitLengthWireCost = gridGraph.getUnitLengthWireCost();
  unitViaCost = gridGraph.getUnitViaCost();
  logisticSlope = parameters.maze_logistic_slope;
  viaMultiplier = parameters.via_multiplier;

  // 初始化hEdgeLengths, vEdgeLengths内存显存
  std::vector<realT> hEdgeLengths(X);
  std::vector<realT> vEdgeLengths(Y);
  hEdgeLengths[0] = vEdgeLengths[0] = 0.f;
  for (int x = 1; x < X; x++)
    hEdgeLengths[x] = gridGraph.getEdgeLength(MetalLayer::H, x - 1);
  for (int y = 1; y < Y; y++)
    vEdgeLengths[y] = gridGraph.getEdgeLength(MetalLayer::V, y - 1);
  devHEdgeLengths = cuda_make_unique<realT[]>(X);
  devVEdgeLengths = cuda_make_unique<realT[]>(Y);
  checkCudaErrors(cudaMemcpy(devHEdgeLengths.get(), hEdgeLengths.data(), X * sizeof(realT), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devVEdgeLengths.get(), vEdgeLengths.data(), Y * sizeof(realT), cudaMemcpyHostToDevice));

  // 初始化layerMinLengths, unitLengthShortCosts内存显存
  std::vector<realT> layerMinLengths(LAYER);
  std::vector<realT> unitLengthShortCosts(LAYER);
  for (int l = 0; l < LAYER; l++)
  {
    layerMinLengths[l] = gridGraph.getLayerMinLength(l);
    unitLengthShortCosts[l] = gridGraph.getUnitLengthShortCost(l);
  }
  devLayerMinLengths = cuda_make_unique<realT[]>(LAYER);
  devUnitLengthShortCosts = cuda_make_unique<realT[]>(LAYER);
  checkCudaErrors(cudaMemcpy(devLayerMinLengths.get(), layerMinLengths.data(), LAYER * sizeof(realT), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devUnitLengthShortCosts.get(), unitLengthShortCosts.data(), LAYER * sizeof(realT), cudaMemcpyHostToDevice));

  // 初始化capacity, demand, wireCost, viaCost显存
  devCapacity = cuda_make_unique<realT[]>(LAYER * N * N);
  devDemand = cuda_make_unique<realT[]>(LAYER * N * N);
  devWireCost = cuda_make_shared<realT[]>(LAYER * N * N);
  devViaCost = cuda_make_shared<realT[]>(LAYER * N * N);
  std::vector<realT> capacity(LAYER * N * N);
  std::vector<realT> demand(LAYER * N * N);
  for (int z = 0; z < LAYER; z++)
  {
    if ((z & 1) ^ DIRECTION)
    {
      for (int y = 0; y < Y; y++)
      {
        for (int x = 1; x < X; x++)
        {
          capacity[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x - 1, y).capacity;
          demand[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x - 1, y).demand;
        }
      }
    }
    else
    {
      for (int x = 0; x < X; x++)
      {
        for (int y = 1; y < Y; y++)
        {
          capacity[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x, y - 1).capacity;
          demand[xyzToIdx(x, y, z, DIRECTION, N)] = gridGraph.getEdge(z, x, y - 1).demand;
        }
      }
    }
  }
  checkCudaErrors(cudaMemcpy(devCapacity.get(), capacity.data(), LAYER * N * N * sizeof(realT), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devDemand.get(), demand.data(), LAYER * N * N * sizeof(realT), cudaMemcpyHostToDevice));

  // 初始化rootIndices, isRouteNet, isOverflowNet内存显存
  rootIndices.resize(numNets);
  std::transform(nets.begin(), nets.end(), rootIndices.begin(), [&](const GRNet &net)
                 { return net.getRoutingTree() == nullptr ? 0 : xyzToIdx(net.getRoutingTree()->x, net.getRoutingTree()->y, net.getRoutingTree()->layerIdx, DIRECTION, N); });
  isRoutedNet.resize(numNets, 0);
  std::transform(nets.begin(), nets.end(), isRoutedNet.begin(), [](const GRNet &net)
                 { return net.getRoutingTree() != nullptr; });
  isOverflowNet.resize(numNets, 0);
  devIsOverflowNet = cuda_make_unique<int[]>(numNets);
  devNetIndices = cuda_make_unique<int[]>(numNets);

  // 初始化allRoutes, allRoutesOffset显存内存
  allRoutesOffset.resize(numNets + 1, 0);
  for (int i = 0; i < numNets; i++)
    allRoutesOffset[i + 1] = allRoutesOffset[i] + nets[i].getNumPins() * MAX_ROUTE_LEN_PER_PIN;
  allRoutes.resize(allRoutesOffset.back(), 0);
  for (int i = 0; i < numNets; i++)
  {
    int *routes = allRoutes.data() + allRoutesOffset[i];
    auto tree = nets[i].getRoutingTree();
    if (tree != nullptr)
    {
      routes[0] = 0;
      GRTreeNode::preorder(tree, [&](std::shared_ptr<GRTreeNode> node)
                           {
        int nodeIdx = xyzToIdx(node->x, node->y, node->layerIdx, DIRECTION, N);
        for(auto child : node->children)
        {
          int childIdx = xyzToIdx(child->x, child->y, child->layerIdx, DIRECTION, N);
          routes[++routes[0]] = std::min(nodeIdx, childIdx);
          routes[++routes[0]] = std::max(nodeIdx, childIdx);
        } });
    }
  }
  devAllRoutesOffset = cuda_make_unique<int[]>(allRoutesOffset.size());
  devAllRoutes = cuda_make_unique<int[]>(allRoutes.size());
  checkCudaErrors(cudaMemcpy(devAllRoutesOffset.get(), allRoutesOffset.data(), allRoutesOffset.size() * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devAllRoutes.get(), allRoutes.data(), allRoutes.size() * sizeof(int), cudaMemcpyHostToDevice));

  // 标记overflow net
  markOverflowNet<<<(numNets + 1023) / 1024, 1024>>>(
      devIsOverflowNet.get(), devDemand.get(), devCapacity.get(), devAllRoutes.get(), devAllRoutesOffset.get(), numNets, DIRECTION, N);
  checkCudaErrors(cudaMemcpy(isOverflowNet.data(), devIsOverflowNet.get(), numNets * sizeof(int), cudaMemcpyDeviceToHost));

  // rotuer
  if(scaleX > 1 || scaleY > 1)
  {
    gridScaler = std::make_unique<GridScaler>(DIRECTION, N, X, Y, LAYER, scaleX, scaleY);
    gridScaler->setWireCostMap(devWireCost);
    gridScaler->setViaCostMap(devViaCost);
    basicGamer = std::make_unique<BasicGamer>(DIRECTION, gridScaler->getCoarseN(), gridScaler->getCoarseX(), gridScaler->getCoarseY(), LAYER, maxNumPins);
    basicGamer->setWireCostMap(gridScaler->getCoarseWireCost());
    basicGamer->setViaCostMap(gridScaler->getCoarseViaCost());
    guidedGamer = std::make_unique<GuidedGamer>(DIRECTION, N, X, Y, LAYER, maxNumPins);
    guidedGamer->setWireCostMap(devWireCost);
    guidedGamer->setViaCostMap(devViaCost);
    int coarseSweepTurns = 5;
    int nWires = std::max(scaleX, scaleY) * (LAYER + 1) * coarseSweepTurns * 2;
    int nRows = ((std::max(scaleX, scaleY) * N + scaleX * scaleY * LAYER) * coarseSweepTurns + PACK_ROW_SIZE - 1) / PACK_ROW_SIZE * 2;
    int nLongWires = std::max(scaleX, scaleY) * coarseSweepTurns * 2;
    int nWorkplace = nLongWires * (N + PACK_ROW_SIZE - 1) / PACK_ROW_SIZE * PACK_ROW_SIZE;
    int nViasegs = scaleX * scaleY * coarseSweepTurns * 2;
    guidedGamer->reserve(nWires, nRows, nLongWires, nWorkplace, nViasegs);
  }
  else
  {
    basicGamer = std::make_unique<BasicGamer>(DIRECTION, N, X, Y, LAYER, maxNumPins);
    basicGamer->setWireCostMap(devWireCost);
    basicGamer->setViaCostMap(devViaCost);
  }
}

void GPUMazeRouter::route(const std::vector<int> &netIndices, int sweepTurns, int margin)
{
  std::vector<int> guide(maxNumPins * MAX_ROUTE_LEN_PER_PIN);
  // 撤销
  checkCudaErrors(cudaMemcpy(devNetIndices.get(), netIndices.data(), netIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
  commitRoutes<<<(netIndices.size() + 1023) / 1024, 1024>>>(
      devDemand.get(), devAllRoutes.get(), devAllRoutesOffset.get(), devNetIndices.get(), numNets, 1, DIRECTION, N, X, Y, LAYER);
  for (int i = 0; i < netIndices.size(); i++)
  {
    int netId = netIndices[i];
    std::vector<int> pinIndices = selectAccessPoints(nets[netId]);
    rootIndices[netId] = pinIndices.front();
    // 扫描范围
    utils::BoxT<int> box = nets[netId].getBoundingBox();
    box.Set(std::max(0, box.lx() - margin), std::max(0, box.ly() - margin),
            std::min(X, box.hx() + margin), std::min(Y, box.hy() + margin));
    // 计算wireCost, viaCost
    calculateWireViaCost<<<dim3((box.width() + 31) / 32, (box.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
        devWireCost.get(), devViaCost.get(), devDemand.get(), devCapacity.get(), devHEdgeLengths.get(), devVEdgeLengths.get(),
        devLayerMinLengths.get(), devUnitLengthShortCosts.get(), unitLengthWireCost, unitViaCost, logisticSlope, viaMultiplier,
        box.lx(), box.ly(), box.width(), box.height(), DIRECTION, N, X, Y, LAYER);
    if(scaleX > 1 || scaleY > 1)
    {
      auto coarseBox = gridScaler->calculateCoarseBoudingBox(box);
      auto coarsePinIndices = gridScaler->calculateCoarsePinIndices(pinIndices);
      gridScaler->scale(coarseBox);
      basicGamer->route(coarsePinIndices, sweepTurns);
      bool coarseIsRouted = basicGamer->getIsRouted();
      if(!coarseIsRouted)
        utils::log() << "gamer error: coarsely route net(id=" << netId << ") failed\n";

      auto devCoarseRoutes = basicGamer->getRoutes();
      checkCudaErrors(cudaMemcpy(guide.data(), devCoarseRoutes.get(), guide.size() * sizeof(int), cudaMemcpyDeviceToHost));
      guidedGamer->setGuide(guide.data(), scaleX, scaleY, gridScaler->getCoarseN());
      guidedGamer->route(pinIndices, sweepTurns + 3);
      bool fineIsRouted = guidedGamer->getIsRouted();
      if(!fineIsRouted)
        utils::log() << "gamer error: finely route net(id=" << netId << ") failed\n";

      isRoutedNet[netId] = fineIsRouted;
      auto devRoutes = guidedGamer->getRoutes();
      checkCudaErrors(cudaMemcpy(devAllRoutes.get() + allRoutesOffset[netId], devRoutes.get(), (allRoutesOffset[netId + 1] - allRoutesOffset[netId]) * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    else
    {
      // route
      basicGamer->route(pinIndices, sweepTurns, box);
      isRoutedNet[netId] = basicGamer->getIsRouted();
      auto devRoutes = basicGamer->getRoutes();
      checkCudaErrors(cudaMemcpy(devAllRoutes.get() + allRoutesOffset[netId], devRoutes.get(), (allRoutesOffset[netId + 1] - allRoutesOffset[netId]) * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    commitRoutes<<<1, 1>>>(
        devDemand.get(), devAllRoutes.get(), devAllRoutesOffset.get(), devNetIndices.get() + i, numNets, 0, DIRECTION, N, X, Y, LAYER);
    if(!isRoutedNet[netId])
      utils::log() << "gamer error: route net(id=" << netId << ") failed\n";
  }
  markOverflowNet<<<(numNets + 1023) / 1024, 1024>>>(
      devIsOverflowNet.get(), devDemand.get(), devCapacity.get(), devAllRoutes.get(), devAllRoutesOffset.get(), numNets, DIRECTION, N);
  checkCudaErrors(cudaMemcpy(isOverflowNet.data(), devIsOverflowNet.get(), numNets * sizeof(int), cudaMemcpyDeviceToHost));
}

void GPUMazeRouter::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  netIndices.clear();
  for (int netId = 0; netId < numNets; netId++)
  {
    if (isOverflowNet[netId])
      netIndices.push_back(netId);
  }
}

void GPUMazeRouter::commit(const std::vector<int> &netIndices)
{
  checkCudaErrors(cudaMemcpy(allRoutes.data(), devAllRoutes.get(), allRoutes.size() * sizeof(int), cudaMemcpyDeviceToHost));
  // TODO: bactching and multi-threading
  for (int netId : netIndices)
  {
    if (isRoutedNet[netId])
    {
      // ripup old tree
      auto oldTree = nets[netId].getRoutingTree();
      if (oldTree != nullptr)
        gridGraph.commitTree(oldTree, true);
      // commit new tree
      auto newTree = extractGRTree(allRoutes.data() + allRoutesOffset[netId], rootIndices[netId], DIRECTION, N);
      nets[netId].setRoutingTree(newTree);
      gridGraph.commitTree(newTree, false);

      // // check if each pin of the net is routed
      // std::unordered_set<int> routePoints;
      // GRTreeNode::preorder(newTree, [&](std::shared_ptr<GRTreeNode> node){
      //   for(const auto &child : node->children)
      //   {
      //     if(child->layerIdx == node->layerIdx)
      //     {
      //       int nodeIdx = xyzToIdx(node->x, node->y, node->layerIdx, DIRECTION, N);
      //       int childIdx = xyzToIdx(child->x, child->y, child->layerIdx, DIRECTION, N);
      //       int startIdx = std::min(nodeIdx, childIdx);
      //       int endIdx = std::max(nodeIdx, childIdx);
      //       for(int idx = startIdx; idx <= endIdx; idx++)
      //         routePoints.insert(idx);
      //     }
      //     else
      //     {
      //       int x = node->x;
      //       int y = node->y;
      //       int startZ = std::min(child->layerIdx, node->layerIdx);
      //       int endZ = std::max(child->layerIdx, node->layerIdx);
      //       for(int z = startZ; z <= endZ; z++)
      //         routePoints.insert(xyzToIdx(x, y, z, DIRECTION, N));
      //     }
      //   }
      // });
      // for(const auto &accessPoints : nets[netId].getPinAccessPoints())
      // {
      //   bool isRoutePin = false;
      //   for(const auto &point : accessPoints)
      //   {
      //     auto idx = xyzToIdx(point.x, point.y, point.layerIdx, DIRECTION, N);
      //     if(routePoints.find(idx) != routePoints.end())
      //     {
      //       isRoutePin = true;
      //       break;
      //     }
      //   }
      //   if(!isRoutePin)
      //   {
      //     log() << "Gamer error: net(id=" << netId << ") is not routed\n";
      //     std::ofstream errorLog(std::to_string(netId) + ".txt");
      //     errorLog << DIRECTION << " " << N << " " << X << " " << Y << " " << LAYER << "\n";
      //     errorLog << "\n";

      //     const int *routes = allRoutes.data() + allRoutesOffset[netId];
      //     for(int i = 0; i < routes[0]; i += 2)
      //       errorLog << routes[1 + i] << " " << routes[2 + i] << "\n";
      //     errorLog << "\n";

      //     for(const auto &accessPoints : nets[netId].getPinAccessPoints())
      //     {
      //       for(const auto &p : accessPoints)
      //         errorLog << xyzToIdx(p.x, p.y, p.layerIdx, DIRECTION, N) << " ";
      //       errorLog << "\n";
      //     }
      //     errorLog << "\n";
      //     break;
      //   }
      // }
    }
    else
    {
      utils::log() << "warning: net(id=" << netId << ") is not routed\n";
    }
  }
}

std::vector<int> GPUMazeRouter::selectAccessPoints(const GRNet &net) const
{
  std::set<int> selectedAccessPoints;
  const auto &boundingBox = net.getBoundingBox();
  utils::PointT<int> netCenter(boundingBox.cx(), boundingBox.cy());
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
      utils::log() << "Warning: the pin is hard to access." << std::endl;
    selectedAccessPoints.insert(xyzToIdx(points[bestIndex].x, points[bestIndex].y, points[bestIndex].layerIdx, DIRECTION, N));
  }
  return std::vector<int>(selectedAccessPoints.begin(), selectedAccessPoints.end());
}

#endif