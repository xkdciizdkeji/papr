#ifdef ENABLE_CUDA

#include "GPURouteContext.cuh"
#include <numeric>
#include <algorithm>
#include <stack>

// cuda kernel
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

__global__ static void markOpenNet(int *isOpenNet, const int *allRoutes, const int *allRoutesOffset, int numNets)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= numNets)
    return;
  const int *routes = allRoutes + allRoutesOffset[nid];
  isOpenNet[nid] = (routes[0] == 0);
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

// -----------------------
// GPURouteContext
// -----------------------

GPURouteContext::GPURouteContext(std::vector<GRNet> &nets, GridGraph &gridGraph, const Parameters &parameters)
    : nets(nets), gridGraph(gridGraph), parameters(parameters)
{
  DIRECTION = gridGraph.getLayerDirection(0) == MetalLayer::H;
  LAYER = gridGraph.getNumLayers();
  X = gridGraph.getSize(0);
  Y = gridGraph.getSize(1);
  N = std::max(X, Y);
  maxNumPins = 0;
  for (const auto &net : nets)
  {
    maxNumPins = std::max(maxNumPins, net.getNumPins());
  }

  // 初始化每个net的access point和boudingbox
  allNetPinIndices.resize(nets.size());
  allNetBoudingBoxes.resize(nets.size());
  for (int netId = 0; netId < nets.size(); netId++)
  {
    selectAccessPoints(nets[netId], allNetPinIndices[netId]);
    allNetBoudingBoxes[netId] = getPinIndicesBoundingBox(allNetPinIndices[netId], DIRECTION, N);
  }
  allNetPinIndicesOffset.resize(nets.size() + 1);
  allNetPinIndicesOffset[0] = 0;
  for(int netId = 1; netId <= nets.size(); netId++)
    allNetPinIndicesOffset[netId] = allNetPinIndicesOffset[netId - 1] + static_cast<int>(allNetPinIndices[netId - 1].size());
  devAllNetPinIndices = cuda_make_shared<int[]>(allNetPinIndicesOffset.back());
  checkCudaErrors(cudaMemcpy(devAllNetPinIndices.get(), allNetPinIndices.data(), allNetPinIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

  // unit cost
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
  for (int z = 0; z < LAYER; z++)
    unitOverflowCosts[z] = gridGraph.getUnitOverflowCost(z);
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

  // 初始化allNetRootIndices, allNetRoutes, allNetRoutesOffset显存内存
  allNetRootIndices.resize(nets.size());
  allNetRoutesOffset.resize(nets.size() + 1, 0);
  for (int netId = 0; netId < nets.size(); netId++)
    allNetRoutesOffset[netId + 1] = allNetRoutesOffset[netId] + nets[netId].getNumPins() * MAX_ROUTE_LEN_PER_PIN;
  allNetRoutes.resize(allNetRoutesOffset.back(), 0);
  for (int netId = 0; netId < nets.size(); netId++)
  {
    int *routes = allNetRoutes.data() + allNetRoutesOffset[netId];
    auto tree = nets[netId].getRoutingTree();
    allNetRootIndices[netId] = 0;;
    routes[0] = 0;
    if (tree != nullptr)
    {
      allNetRootIndices[netId] = xyzToIdx(tree->x, tree->y, tree->layerIdx, DIRECTION, N);
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
  devAllNetRoutesOffset = cuda_make_unique<int[]>(allNetRoutesOffset.size());
  devAllNetRoutes = cuda_make_unique<int[]>(allNetRoutes.size());
  checkCudaErrors(cudaMemcpy(devAllNetRoutesOffset.get(), allNetRoutesOffset.data(), allNetRoutesOffset.size() * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devAllNetRoutes.get(), allNetRoutes.data(), allNetRoutes.size() * sizeof(int), cudaMemcpyHostToDevice));
}

void GPURouteContext::commit(int netId, int rootIdx, const cuda_shared_ptr<int[]> &routes)
{
  allNetRootIndices[netId] = rootIdx;
  commitRoutes<<<1, 1>>>(devDemand.get(), devFlag.get(), routes.get(), 0, DIRECTION, N, X, Y, LAYER);
  checkCudaErrors(cudaMemcpy(devAllNetRoutes.get() + allNetRoutesOffset[netId], routes.get(),
                             (allNetRoutesOffset[netId + 1] - allNetRoutesOffset[netId]) * sizeof(int), cudaMemcpyDeviceToDevice));
}

void GPURouteContext::reverse(int netId)
{
  commitRoutes<<<1, 1>>>(devDemand.get(), devFlag.get(), devAllNetRoutes.get() + allNetRoutesOffset[netId], 1, DIRECTION, N, X, Y, LAYER);
}

void GPURouteContext::updateCost(const utils::BoxT<int> &box)
{
  calculateWireViaCost<<<dim3((box.width() + 31) / 32, (box.height() + 31) / 32, LAYER), dim3(32, 32, 1)>>>(
      devWireCost.get(), devNonStackViaCost.get(), devDemand.get(), devCapacity.get(),
      devHEdgeLengths.get(), devVEdgeLengths.get(), devUnitOverflowCosts.get(), unitLengthWireCost, unitViaCost,
      box.lx(), box.ly(), box.width(), box.height(), DIRECTION, N, X, Y, LAYER);
}

void GPURouteContext::getOverflowAndOpenNetIndices(std::vector<int> &netIndices) const
{
  auto devIsOverflowNet = cuda_make_unique<int[]>(nets.size());
  markOverflowNet<<<(nets.size() + 1023) / 1024, 1024>>>(
      devIsOverflowNet.get(), devDemand.get(), devCapacity.get(), devAllNetRoutes.get(), devAllNetRoutesOffset.get(), nets.size(), DIRECTION, N);
  auto isOverflowNet = std::make_unique<int[]>(nets.size());
  checkCudaErrors(cudaMemcpy(isOverflowNet.get(), devIsOverflowNet.get(), nets.size() * sizeof(int), cudaMemcpyDeviceToHost));

  auto devIsOpenNet = cuda_make_unique<int[]>(nets.size());
  markOpenNet<<<(nets.size() + 1023) / 1024, 1024>>>(devIsOpenNet.get(), devAllNetRoutes.get(), devAllNetRoutesOffset.get(), nets.size());
  auto isOpenNet = std::make_unique<int[]>(nets.size());
  checkCudaErrors(cudaMemcpy(isOpenNet.get(), devIsOpenNet.get(), nets.size() * sizeof(int), cudaMemcpyDeviceToHost));

  netIndices.clear();
  for (int netId = 0; netId < nets.size(); netId++)
  {
    if (isOpenNet[netId] || isOverflowNet[netId])
      netIndices.push_back(netId);
  }
}

void GPURouteContext::applyToCpu(const std::vector<int> &netIndices)
{
  checkCudaErrors(cudaMemcpy(allNetRoutes.data(), devAllNetRoutes.get(), allNetRoutes.size() * sizeof(int), cudaMemcpyDeviceToHost));
  for (int netId : netIndices)
  {
    // printf("apply net(id=%d)\n", netId);
    auto oldTree = nets[netId].getRoutingTree();
    if (oldTree != nullptr)
      gridGraph.commitTree(oldTree, true);
    // commit new tree
    auto newTree = extractGRTree(allNetRoutes.data() + allNetRoutesOffset[netId], allNetRootIndices[netId]);
    nets[netId].setRoutingTree(newTree);
    gridGraph.commitTree(newTree, false);

    // checkNetIsRouted(nets[netId]);
  }
}

std::shared_ptr<GRTreeNode> GPURouteContext::extractGRTree(const int *routes, int rootIdx) const
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

bool GPURouteContext::checkNetIsRouted(const GRNet &net) const
{
  // check if each pin of the net is routed
  auto tree = net.getRoutingTree();
  std::unordered_set<int> points;
  points.insert(xyzToIdx(tree->x, tree->y, tree->layerIdx, DIRECTION, N));
  auto collect = [&](std::shared_ptr<GRTreeNode> node)
  {
    for (const auto &child : node->children)
    {
      if (child->layerIdx == node->layerIdx)
      {
        int nodeIdx = xyzToIdx(node->x, node->y, node->layerIdx, DIRECTION, N);
        int childIdx = xyzToIdx(child->x, child->y, child->layerIdx, DIRECTION, N);
        int startIdx = std::min(nodeIdx, childIdx);
        int endIdx = std::max(nodeIdx, childIdx);
        for (int idx = startIdx; idx <= endIdx; idx++)
          points.insert(idx);
      }
      else
      {
        int x = node->x;
        int y = node->y;
        int startZ = std::min(child->layerIdx, node->layerIdx);
        int endZ = std::max(child->layerIdx, node->layerIdx);
        for (int z = startZ; z <= endZ; z++)
          points.insert(xyzToIdx(x, y, z, DIRECTION, N));
      }
    }
  };
  GRTreeNode::preorder(tree, collect);

  for (const auto &accessPoints : net.getPinAccessPoints())
  {
    bool isRoutePin = false;
    for (const auto &point : accessPoints)
    {
      auto idx = xyzToIdx(point.x, point.y, point.layerIdx, DIRECTION, N);
      if (points.find(idx) != points.end())
      {
        isRoutePin = true;
        break;
      }
    }
    if (!isRoutePin)
    {
      log() << "Gamer error: net(id=" << net.getIndex() << ") is not routed\n";
      return false;
      // std::ofstream errorLog(std::to_string(netId) + ".txt");
      // errorLog << DIRECTION << " " << N << " " << X << " " << Y << " " << LAYER << "\n";
      // errorLog << "\n";

      // const int *routes = allRoutes.data() + allRoutesOffset[netId];
      // for(int i = 0; i < routes[0]; i += 2)
      //   errorLog << routes[1 + i] << " " << routes[2 + i] << "\n";
      // errorLog << "\n";

      // for(const auto &accessPoints : nets[netId].getPinAccessPoints())
      // {
      //   for(const auto &p : accessPoints)
      //     errorLog << xyzToIdx(p.x, p.y, p.layerIdx, DIRECTION, N) << " ";
      //   errorLog << "\n";
      // }
      // errorLog << "\n";
    }
  }
  return true;
}

void GPURouteContext::selectAccessPoints(const GRNet &net, std::vector<int> &pinIndices) const
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

utils::BoxT<int> GPURouteContext::getPinIndicesBoundingBox(const std::vector<int> &pinIndices, int DIRECTION, int N) const
{
  utils::BoxT<int> box;
  for (int idx : pinIndices)
  {
    auto [x, y, z] = idxToXYZ(idx, DIRECTION, N);
    box.Update(x, y);
  }
  return box;
}

#endif