#ifdef ENABLE_CUDA
#include "GPUMazeRouter.cuh"
#include <numeric>
#include <stack>

// ----------------------------
// Cuda kernel
// ----------------------------

// __global__ static void commitRoutes(realT *demand, const int *allRoutes, const int *allRoutesOffset, const int *netIndices,
//                                     int reverse, int numNets, int DIRECTION, int N, int X, int Y, int LAYER)
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= numNets)
//     return;
//   const int *routes = allRoutes + allRoutesOffset[netIndices[tid]];
//   for (int i = 0; i < routes[0]; i += 2)
//   {
//     int startIdx = routes[1 + i];
//     int endIdx = routes[2 + i];
//     auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
//     auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
//     if (startZ == endZ) // wire
//     {
//       for (int idx = startIdx + 1; idx <= endIdx; idx++)
//       {
// #if __CUDA_ARCH__ >= 600
//         atomicAdd(demand + idx, reverse ? static_cast<realT>(-1) : static_cast<realT>(1));
// #else
//         myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
// #endif
//       }
//     }
//     else // vias
//     {
//       for (int z = startZ + 1, x = startX, y = startY; z <= endZ - 1; z++)
//       {
//         int idx = xyzToIdx(x, y, z, DIRECTION, N);
// #if __CUDA_ARCH__ >= 600
//         if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
//           atomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
//         else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
//           atomicAdd(demand + idx, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
//         else
//         {
//           atomicAdd(demand + idx, reverse ? static_cast<realT>(-.5f) : static_cast<realT>(.5f));
//           atomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-.5f) : static_cast<realT>(.5f));
//         }
// #else
//         if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
//           myAtomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
//         else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
//           myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-1.f) : static_cast<realT>(1.f));
//         else
//         {
//           myAtomicAdd(demand + idx, reverse ? static_cast<realT>(-.5f) : static_cast<realT>(.5f));
//           myAtomicAdd(demand + idx + 1, reverse ? static_cast<realT>(-.5f) : static_cast<realT>(.5f));
//         }
// #endif
//       }
//     }
//   }
// }

__global__ static void commitRoutes(realT *demand, int *flag, const int *routes,
                                    int reverse, int DIRECTION, int N, int X, int Y, int LAYER)
{
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ == endZ)
    {
      for (int idx = startIdx; idx <= endIdx; idx++)
        flag[idx] = 0;
    }
    else
    {
      for (int z = startZ; z <= endZ; z++)
        flag[xyzToIdx(startX, startY, z, DIRECTION, N)] = 0;
    }
  }
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ == endZ)
    {
      for (int idx = startIdx; idx <= endIdx; idx++)
      {
        flag[idx] = 1;
        if (idx > startIdx)
          demand[idx] += reverse ? -1.f : 1.f;
      }
    }
  }
  for (int i = 0; i < routes[0]; i += 2)
  {
    int startIdx = routes[1 + i];
    int endIdx = routes[2 + i];
    auto [startX, startY, startZ] = idxToXYZ(startIdx, DIRECTION, N);
    auto [endX, endY, endZ] = idxToXYZ(endIdx, DIRECTION, N);
    if (startZ != endZ)
    {
      for (int z = startZ, x = startX, y = startY; z < endZ; z++)
      {
        int idx = xyzToIdx(startX, startY, z, DIRECTION, N);
        if (!flag[idx])
        {
          flag[idx] = 1;
          if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
            demand[idx + 1] += reverse ? -1.f : 1.f;
          else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
            demand[idx] += reverse ? -1.f : 1.f;
          else
          {
            demand[idx] += reverse ? -.5f : .5f;
            demand[idx + 1] += reverse ? -.5f : .5f;
          }
        }
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

__global__ static void calculateWireViaCost(realT *wireCost, realT *nonStackViaCost, const realT *demand, const realT *capacity,
                                            const realT *hEdgeLengths, const realT *vEdgeLengths,
                                            const realT *unitOverflowCosts, realT unitLengthWireCost, realT unitViaCost,
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
  realT slope = capacity[idx] > 0.f ? 0.5f : 1.5f;
  wireCost[idx] = edgeLength * unitLengthWireCost + unitOverflowCosts[z] * exp(slope * (demand[idx] - capacity[idx])) * (exp(slope) - 1);

  // via cost
  if ((z & 1) ^ DIRECTION ? (x == 0) : (y == 0))
  {
    realT rs = capacity[idx + 1] > 0.f ? 0.5f : 1.5f;
    nonStackViaCost[idx] = unitOverflowCosts[z] * exp(rs * (demand[idx + 1] - capacity[idx + 1])) * (exp(rs) - 1);
  }
  else if ((z & 1) ^ DIRECTION ? (x == X - 1) : (y == Y - 1))
  {
    realT ls = capacity[idx] > 0.f ? 0.5f : 1.5f;
    nonStackViaCost[idx] = unitOverflowCosts[z] * exp(ls * (demand[idx] - capacity[idx])) * (exp(ls) - 1);
  }
  else
  {
    realT ls = capacity[idx] > 0.f ? 0.5f : 1.5f;
    realT rs = capacity[idx + 1] > 0.f ? 0.5f : 1.5f;
    nonStackViaCost[idx] = unitOverflowCosts[z] * exp(ls * (demand[idx] - capacity[idx])) * (exp(ls * 0.5f) - 1) +
                           unitOverflowCosts[z] * exp(rs * (demand[idx + 1] - capacity[idx + 1])) * (exp(rs * 0.5f) - 1);
  }
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
  maxNumPins = std::transform_reduce(
      nets.begin(), nets.end(), 0, [](int x, int y)
      { return std::max(x, y); },
      [](const GRNet &net)
      { return net.getNumPins(); });

  unitLengthWireCost = gridGraph.getUnitLengthWireCost();
  unitViaCost = gridGraph.getUnitViaCost();

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

  // 初始化unitOverflowCosts内存显存
  std::vector<realT> unitOverflowCosts(LAYER);
  for (int l = 0; l < LAYER; l++)
    unitOverflowCosts[l] = 50*(gridGraph.getUnitOverflowCost(l));
  devUnitOverflowCosts = cuda_make_unique<realT[]>(LAYER);
  checkCudaErrors(cudaMemcpy(devUnitOverflowCosts.get(), unitOverflowCosts.data(), LAYER * sizeof(realT), cudaMemcpyHostToDevice));

  // 初始化flag, capacity, demand, wireCost, viaCost显存
  devFlag = cuda_make_unique<int[]>(LAYER * N * N);
  devCapacity = cuda_make_unique<realT[]>(LAYER * N * N);
  devDemand = cuda_make_unique<realT[]>(LAYER * N * N);
  devWireCost = cuda_make_shared<realT[]>(LAYER * N * N);
  devNonStackViaCost = cuda_make_shared<realT[]>(LAYER * N * N);
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
  rootIndices.resize(nets.size());
  std::transform(nets.begin(), nets.end(), rootIndices.begin(), [&](const GRNet &net)
                 { return net.getRoutingTree() == nullptr ? 0 : xyzToIdx(net.getRoutingTree()->x, net.getRoutingTree()->y, net.getRoutingTree()->layerIdx, DIRECTION, N); });
  isRoutedNet.resize(nets.size(), 0);
  std::transform(nets.begin(), nets.end(), isRoutedNet.begin(), [](const GRNet &net)
                 { return net.getRoutingTree() != nullptr; });
  isOverflowNet.resize(nets.size(), 0);
  devIsOverflowNet = cuda_make_unique<int[]>(nets.size());

  // 初始化allRoutes, allRoutesOffset显存内存
  allRoutesOffset.resize(nets.size() + 1, 0);
  for (int i = 0; i < nets.size(); i++)
    allRoutesOffset[i + 1] = allRoutesOffset[i] + nets[i].getNumPins() * MAX_ROUTE_LEN_PER_PIN;
  allRoutes.resize(allRoutesOffset.back(), 0);
  for (int i = 0; i < nets.size(); i++)
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

  // // onestep pipeline init
  // gamer = std::make_unique<BasicGamer>(DIRECTION, N, X, Y, LAYER, maxNumPins);
  // gamer->setWireCost(devWireCost);
  // gamer->setNonStackViaCost(devNonStackViaCost);
  // gamer->setUnitViaCost(unitViaCost);

  // twostep pipeline init
  int scaleX = std::min(X / 256, MAX_SCALE);
  int scaleY = std::min(Y / 256, MAX_SCALE);
  extractor2D = std::make_unique<Grid2DExtractor>(DIRECTION, N, X, Y, LAYER);
  extractor2D->setWireCost(devWireCost);
  scaler2D = std::make_unique<GridScaler2D>(X, Y, scaleX, scaleY);
  scaler2D->setCost2D(extractor2D->getCost2D());
  coarseGamer2D = std::make_unique<BasicGamer2D>(scaler2D->getCoarseX(), scaler2D->getCoarseY(), maxNumPins);
  coarseGamer2D->setCost2D(scaler2D->getCoarseCost2D());
  fineGamer = std::make_unique<GuidedGamer>(DIRECTION, N, X, Y, LAYER, maxNumPins);
  fineGamer->setWireCost(devWireCost);
  fineGamer->setNonStackViaCost(devNonStackViaCost);
  fineGamer->setUnitViaCost(unitViaCost);
}

// void GPUMazeRouter::route(const std::vector<int> &netIndices, int numTurns, int margin)
// {
//   // rip-up
//   checkCudaErrors(cudaMemcpy(devNetIndices.get(), netIndices.data(), netIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
//   commitRoutes<<<(netIndices.size() + 1023) / 1024, 1024>>>(
//       devDemand.get(), devAllRoutes.get(), devAllRoutesOffset.get(), devNetIndices.get(), 1, netIndices.size(), DIRECTION, N, X, Y, LAYER);
//   // route
//   std::vector<int> pinIndices(maxNumPins);
//   for (int i = 0; i < netIndices.size(); i++)
//   {
//     int netId = netIndices[i];
//     selectAccessPoints(nets[netId], pinIndices);
//     utils::BoxT<int> box = getPinIndicesBoundingBox(pinIndices, DIRECTION, N);
//     box.Set(std::max(0, box.lx() - margin), std::max(0, box.ly() - margin),
//             std::min(X, box.hx() + margin), std::min(Y, box.hy() + margin));
//     rootIndices[netId] = pinIndices.front();
//     calculateWireViaCost<<<dim3((box.width() + 31) / 32, (box.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
//         devWireCost.get(), devNonStackViaCost.get(), devDemand.get(), devCapacity.get(),
//         devHEdgeLengths.get(), devVEdgeLengths.get(), devUnitOverflowCosts.get(), unitLengthWireCost, unitViaCost,
//         box.lx(), box.ly(), box.width(), box.height(), DIRECTION, N, X, Y, LAYER);
//     gamer->route(pinIndices, numTurns, box);
//     isRoutedNet[netId] = gamer->getIsRouted();
//     if (!isRoutedNet[netId])
//     {
//       utils::log() << "gamer error: route net(id=" << netId << ") failed\n";
//       return;
//     }
//     checkCudaErrors(cudaMemcpy(devAllRoutes.get() + allRoutesOffset[netId], gamer->getRoutes().get(),
//                                (allRoutesOffset[netId + 1] - allRoutesOffset[netId]) * sizeof(int), cudaMemcpyDeviceToDevice));
//     commitRoutes<<<1, 1>>>(
//         devDemand.get(), devAllRoutes.get(), devAllRoutesOffset.get(), devNetIndices.get() + i, 0, 1, DIRECTION, N, X, Y, LAYER);
//   }
//   // mark overflow net
//   markOverflowNet<<<(nets.size() + 1023) / 1024, 1024>>>(
//       devIsOverflowNet.get(), devDemand.get(), devCapacity.get(), devAllRoutes.get(), devAllRoutesOffset.get(), nets.size(), DIRECTION, N);
//   checkCudaErrors(cudaMemcpy(isOverflowNet.data(), devIsOverflowNet.get(), nets.size() * sizeof(int), cudaMemcpyDeviceToHost));
// }

void GPUMazeRouter::routeTwoStep(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int coarseMargin)
{
  // reverse
  for (auto netId : netIndices)
  {
    commitRoutes<<<1, 1>>>(
      devDemand.get(), devFlag.get(), devAllRoutes.get() + allRoutesOffset[netId],
      1, DIRECTION, N, X, Y, LAYER);
  }
  // route
  std::vector<int> pinIndices(maxNumPins);
  std::vector<int> pin2DIndices(maxNumPins);
  std::vector<int> coarsePin2DIndices(maxNumPins);
  std::vector<int> routes2D(maxNumPins * MAX_ROUTE_LEN_PER_PIN);
  std::vector<utils::BoxT<int>> guides2D;
  for (int i = 0; i < netIndices.size(); i++)
  {
    int netId = netIndices[i];
    selectAccessPoints(nets[netId], pinIndices);
    rootIndices[netId] = pinIndices.front();
    // compute guides
    guides2D.clear();
    utils::BoxT<int> box = getPinIndicesBoundingBox(pinIndices, DIRECTION, N);
    utils::BoxT<int> coarseBox = scaler2D->coarsenBoudingBox(box);
    coarseBox.Set(
        std::max(0, coarseBox.lx() - coarseMargin), std::max(0, coarseBox.ly() - coarseMargin),
        std::min(scaler2D->getCoarseX(), coarseBox.hx() + coarseMargin), std::min(scaler2D->getCoarseY(), coarseBox.hy() + coarseMargin));
    utils::BoxT<int> fineBox = scaler2D->finingBoundingBox(coarseBox);
    calculateWireViaCost<<<dim3((fineBox.width() + 31) / 32, (fineBox.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
        devWireCost.get(), devNonStackViaCost.get(), devDemand.get(), devCapacity.get(),
        devHEdgeLengths.get(), devVEdgeLengths.get(), devUnitOverflowCosts.get(), unitLengthWireCost, unitViaCost,
        fineBox.lx(), fineBox.ly(), fineBox.width(), fineBox.height(), DIRECTION, N, X, Y, LAYER);
    if (box.width() < 256 && box.height() < 256 && box.width() < scaler2D->getScaleX() && box.height() < scaler2D->getScaleY())
      guides2D.push_back(fineBox);
    else
    {
      extractor2D->extractCost2D(fineBox);
      extractor2D->extractPin2DIndices(pin2DIndices, pinIndices);
      scaler2D->scaleCost2D(coarseBox);
      scaler2D->scalePin2DIndices(coarsePin2DIndices, pin2DIndices);
      coarseGamer2D->route(coarsePin2DIndices, numCoarseTurns, coarseBox);
      auto devRoutes2D = coarseGamer2D->getRoutes2D();
      checkCudaErrors(cudaMemcpy(routes2D.data(), devRoutes2D.get(), routes2D.size() * sizeof(int), cudaMemcpyDeviceToHost));
      scaler2D->getGuideFromRoutes2D(guides2D, routes2D.data());
    }

    // if (netId == 26)
    // {
    //   for(int i = 0; i < guides2D.size(); i++)
    //   {
    //     const auto b = guides2D[i];
    //     //    <mxCell id="1hsR5Wi1GT06HaeS9luZ-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=none;" vertex="1" parent="1">
    //     //      <mxGeometry x="16" y="216" width="70" height="3" as="geometry" />
    //     //    </mxCell>
    //     printf("<mxCell id=\"guide-%d\" value=\"\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=none;\" vertex=\"1\" parent=\"1\">\n", i);
    //     printf("<mxGeometry x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" as=\"geometry\" />\n", b.lx(), b.ly(), b.width(), b.height());
    //     printf("</mxCell>\n");
    //   }
    //   for (const int idx : pinIndices)
    //   {
    //     auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
    //     printf("<mxCell id=\"pin-%d\" value=\"\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#FF0000;strokeColor=none;\" vertex=\"1\" parent=\"1\">\n", idx);
    //     printf("<mxGeometry x=\"%d\" y=\"%d\" width=\"1\" height=\"1\" as=\"geometry\" />\n", x, y);
    //     printf("</mxCell>\n");
    //   }
    // }

    // pinIndices的顺序必须与coarsePin2DIndices的一致
    std::sort(pinIndices.begin(), pinIndices.end(), [&](int idx1, int idx2)
              {
      auto [x1, y1, z1] = idxToXYZ(idx1, DIRECTION, N);
      auto [x2, y2, z2] = idxToXYZ(idx2, DIRECTION, N);
      int rank1 = (x1 / scaler2D->getScaleX()) + (y1 / scaler2D->getScaleY()) * scaler2D->getCoarseX();
      int rank2 = (x2 / scaler2D->getScaleX()) + (y2 / scaler2D->getScaleY()) * scaler2D->getCoarseX();
      return rank1 < rank2; });

    // fine routing using guides
    fineGamer->setGuide2D(guides2D);
    fineGamer->route(pinIndices, numFineTurns);
    isRoutedNet[netId] = fineGamer->getIsRouted();
    if (!isRoutedNet[netId])
    {
      utils::log() << "gamer error: route net(id=" << netId << ") failed\n";
    }
    checkCudaErrors(cudaMemcpy(devAllRoutes.get() + allRoutesOffset[netId], fineGamer->getRoutes().get(),
                               (allRoutesOffset[netId + 1] - allRoutesOffset[netId]) * sizeof(int), cudaMemcpyDeviceToDevice));
    commitRoutes<<<1, 1>>>(devDemand.get(), devFlag.get(), fineGamer->getRoutes().get(), 0, DIRECTION, N, X, Y, LAYER);
  }
}

void GPUMazeRouter::getOverflowNetIndices(std::vector<int> &netIndices)
{
  // mark overflow net
  markOverflowNet<<<(nets.size() + 1023) / 1024, 1024>>>(
      devIsOverflowNet.get(), devDemand.get(), devCapacity.get(), devAllRoutes.get(), devAllRoutesOffset.get(), nets.size(), DIRECTION, N);
  checkCudaErrors(cudaMemcpy(isOverflowNet.data(), devIsOverflowNet.get(), nets.size() * sizeof(int), cudaMemcpyDeviceToHost));
  netIndices.clear();
  for (int netId = 0; netId < nets.size(); netId++)
  {
    if (isOverflowNet[netId])
      netIndices.push_back(netId);
  }
}

void GPUMazeRouter::applyToCpu(const std::vector<int> &netIndices)
{
  checkCudaErrors(cudaMemcpy(allRoutes.data(), devAllRoutes.get(), allRoutes.size() * sizeof(int), cudaMemcpyDeviceToHost));

  // std::ofstream errLog("extract_error.log");
  // errLog << DIRECTION << " " << N << " " << X << " " << Y << " " << LAYER << "\n\n";
  // for (int netId : netIndices)
  // {
  //   const int *routes = allRoutes.data() + allRoutesOffset[netId];
  //   for(int i = 0; i < routes[0]; i += 2)
  //     errLog << routes[1 + i] << " " << routes[2 + i] << "\n";
  //   errLog << "\n";
  // }
  // errLog.close();

  // {
  //   auto netId = 26;
  //   utils::log() << "routes for net(id=" << netId << ")\n";
  //   auto [rootX, rootY, rootZ] = idxToXYZ(rootIndices[netId], DIRECTION, N);
  //   printf("rootIdx: (%d, %d, %d)\n", rootX, rootY, rootZ);
  //   const int *routes = allRoutes.data() + allRoutesOffset[netId];
  //   for(int i = 0; i < routes[0]; i += 2)
  //   {
  //     auto [startX, startY, startZ] = idxToXYZ(routes[1 + i], DIRECTION, N);
  //     auto [endX, endY, endZ] = idxToXYZ(routes[2 + i], DIRECTION, N);
  //     printf("(%d, %d, %d) -> (%d, %d, %d)\n", startX, startY, startZ, endX, endY, endZ);
  //   }
  // }

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
      auto newTree = extractGRTree(allRoutes.data() + allRoutesOffset[netId], rootIndices[netId]);
      nets[netId].setRoutingTree(newTree);
      gridGraph.commitTree(newTree, false);

      // // check if each pin of the net is routed
      // std::unordered_set<int> routePoints;
      // GRTreeNode::preorder(newTree, [&](std::shared_ptr<GRTreeNode> node)
      //                      {
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
      //   } });
      // for (const auto &accessPoints : nets[netId].getPinAccessPoints())
      // {
      //   bool isRoutePin = false;
      //   for (const auto &point : accessPoints)
      //   {
      //     auto idx = xyzToIdx(point.x, point.y, point.layerIdx, DIRECTION, N);
      //     if (routePoints.find(idx) != routePoints.end())
      //     {
      //       isRoutePin = true;
      //       break;
      //     }
      //   }
      //   if (!isRoutePin)
      //   {
      //     log() << "Gamer error: net(id=" << netId << ") is not routed\n";
      //     // std::ofstream errorLog(std::to_string(netId) + ".txt");
      //     // errorLog << DIRECTION << " " << N << " " << X << " " << Y << " " << LAYER << "\n";
      //     // errorLog << "\n";

      //     // const int *routes = allRoutes.data() + allRoutesOffset[netId];
      //     // for(int i = 0; i < routes[0]; i += 2)
      //     //   errorLog << routes[1 + i] << " " << routes[2 + i] << "\n";
      //     // errorLog << "\n";

      //     // for(const auto &accessPoints : nets[netId].getPinAccessPoints())
      //     // {
      //     //   for(const auto &p : accessPoints)
      //     //     errorLog << xyzToIdx(p.x, p.y, p.layerIdx, DIRECTION, N) << " ";
      //     //   errorLog << "\n";
      //     // }
      //     // errorLog << "\n";
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

void GPUMazeRouter::selectAccessPoints(const GRNet &net, std::vector<int> &pinIndices) const
{
  pinIndices.clear();
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
    pinIndices.push_back(xyzToIdx(points[bestIndex].x, points[bestIndex].y, points[bestIndex].layerIdx, DIRECTION, N));
  }
  std::sort(pinIndices.begin(), pinIndices.end());
  auto last = std::unique(pinIndices.begin(), pinIndices.end());
  pinIndices.erase(last, pinIndices.end());
}

utils::BoxT<int> GPUMazeRouter::getPinIndicesBoundingBox(const std::vector<int> &pinIndices, int DIRECTION, int N) const
{
  utils::BoxT<int> box;
  for (int idx : pinIndices)
  {
    auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
    box.Update(x, y);
  }
  return box;
}

std::shared_ptr<GRTreeNode> GPUMazeRouter::extractGRTree(const int *routes, int rootIdx) const
{
  auto [rootX, rootY, rootZ] = idxToXYZ(rootIdx, DIRECTION, N);
  auto root = std::make_shared<GRTreeNode>(rootZ, rootX, rootY);
  if (routes[0] < 2)
    return root;

  auto hash = [&](const int3 &p)
  {
    return std::hash<int>{}(xyzToIdx(p.x, p.y, p.z, DIRECTION, N));
  };
  auto equal = [&](const int3 &p, const int3 &q)
  {
    return p.x == q.x && p.y == q.y && p.z == q.z;
  };
  auto compw = [&](const int3 &p, const int3 &q)
  {
    if (p.z == q.z)
    {
      if ((p.z & 1) ^ DIRECTION) // Y-X
        return (p.y < q.y) || (p.y == q.y && p.x < q.x);
      else
        return (p.x < q.x) || (p.x == q.x && p.y < q.y);
    }
    else
      return p.z < q.z;
  };
  auto compv = [&](const int3 &p, const int3 &q)
  {
    return (p.x < q.x) || (p.x == q.x && p.y < q.y) || (p.x == q.x && p.y == q.y && p.z < q.z);
  };
  // 收集所有交点
  std::vector<int3> allpoints;
  for (int i = 0; i < routes[0]; i += 2)
  {
    allpoints.push_back(idxToXYZ(routes[1 + i], DIRECTION, N));
    allpoints.push_back(idxToXYZ(routes[2 + i], DIRECTION, N));
  }
  std::sort(allpoints.begin(), allpoints.end(), compw);
  auto last = std::unique(allpoints.begin(), allpoints.end(), equal);
  allpoints.erase(last, allpoints.end());
  // 根据交点拆分线段
  std::vector<std::pair<int3, int3>> segments;
  for (int i = 0; i < routes[0]; i += 2)
  {
    int3 start = idxToXYZ(routes[1 + i], DIRECTION, N);
    int3 end = idxToXYZ(routes[2 + i], DIRECTION, N);
    if (start.z == end.z)
    {
      auto startIt = std::lower_bound(allpoints.begin(), allpoints.end(), start, compw);
      auto endIt = std::upper_bound(allpoints.begin(), allpoints.end(), end, compw);
      for (auto it = startIt, nextIt = startIt + 1; nextIt != endIt; it++, nextIt++)
        segments.emplace_back(*it, *nextIt);
    }
  }
  std::sort(allpoints.begin(), allpoints.end(), compv);
  for (int i = 0; i < routes[0]; i += 2)
  {
    int3 start = idxToXYZ(routes[1 + i], DIRECTION, N);
    int3 end = idxToXYZ(routes[2 + i], DIRECTION, N);
    if (start.z != end.z)
    {
      auto startIt = std::lower_bound(allpoints.begin(), allpoints.end(), start, compv);
      auto endIt = std::upper_bound(allpoints.begin(), allpoints.end(), end, compv);
      for (auto it = startIt, nextIt = it + 1; nextIt != endIt; it++, nextIt++)
        segments.emplace_back(*it, *nextIt);
    }
  }
  // 建立连接
  std::unordered_map<int3, std::vector<int3>, decltype(hash), decltype(equal)> linkGraph(allpoints.size(), hash, equal);
  for (const auto &[s, e] : segments)
  {
    linkGraph[s].push_back(e);
    linkGraph[e].push_back(s);
  }
  // extract GRTree using `linkGraph`
  std::unordered_map<int3, bool, decltype(hash), decltype(equal)> mark(allpoints.size(), hash, equal);
  for (const auto &[p, link] : linkGraph)
    mark.emplace(p, false);
  std::stack<std::shared_ptr<GRTreeNode>> stack;
  stack.push(root);
  mark.at(idxToXYZ(rootIdx, DIRECTION, N)) = true;
  while (!stack.empty())
  {
    auto node = stack.top();
    stack.pop();
    const auto &link = linkGraph.at(make_int3(node->x, node->y, node->layerIdx));
    for (const int3 &q : link)
    {
      if (mark.at(q) == false)
      {
        auto child = std::make_shared<GRTreeNode>(q.z, q.x, q.y);
        node->children.push_back(child);
        stack.push(child);
        mark.at(q) = true;
      }
    }
  }
  return root;
}
#endif