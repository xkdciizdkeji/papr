#ifdef ENABLE_CUDA
#include "GuidedGamer.cuh"
#include <numeric>

// ------------------------------
// cuda kernels
// ------------------------------

__global__ static void clearCostAtRow(realT *costAtRow, int numRows)
{
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < numRows * PACK_ROW_SIZE)
    costAtRow[pos] = INFINITY_DISTANCE;
}

__global__ static void clearCostAtViaseg(realT *costAtViaseg, int numViasegs)
{
  int loc = blockIdx.x * blockDim.x + threadIdx.x;
  if (loc < numViasegs * VIA_SEG_SIZE)
    costAtViaseg[loc] = INFINITY_DISTANCE;
}

__global__ static void packWireToRows(int *idxAtRow, realT *costAtRow, int *idxPosMap,
                                      const int3 *packPlan, const realT *wireCost, int numWires, int DIRECTION, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numWires)
    return;
  auto [offset, startIdx, endIdx] = packPlan[tid];
  int z = startIdx / N / N;
  for (int idx = startIdx, i = 0; idx <= endIdx; idx++, i++)
  {
    idxPosMap[idx] = offset + i;
    idxAtRow[offset + i] = idx;
    costAtRow[offset + i] = ((i == 0 || z == 0) ? INFINITY_DISTANCE : wireCost[idx]);
  }
}

__global__ static void packViaToSegs(realT *costAtViaseg, int *posAtViaseg, int *locAtRow, const int3 *packPlan,
                                     const realT *viaCost, const int *idxPosMap, int numViasegs, int DIRECTION, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numViasegs)
    return;
  auto [offset, startIdx, endIdx] = packPlan[tid];
  auto [x, y, startZ] = idxToXYZ(startIdx, DIRECTION, N);
  int endZ = endIdx / N / N;
  for (int z = startZ, i = 0; z <= endZ; z++, i++)
  {
    int idx = xyzToIdx(x, y, z, DIRECTION, N);
    int pos = idxPosMap[idx];
    locAtRow[pos] = offset + i;
    posAtViaseg[offset + i] = pos;
    costAtViaseg[offset + i] = (i == 0 ? INFINITY_DISTANCE : viaCost[idx]);
  }
}

__global__ static void sweepWireShared(realT *distAtRow, int *prevAtRow, const realT *costAtRow, int longWireEndRowsOffset)
{
  __shared__ char shared[4 * PACK_ROW_SIZE * sizeof(realT) + 2 * PACK_ROW_SIZE * sizeof(int)];
  realT *cL = (realT *)(shared);
  realT *cR = (realT *)(cL + PACK_ROW_SIZE);
  realT *dL = (realT *)(cR + PACK_ROW_SIZE);
  realT *dR = (realT *)(dL + PACK_ROW_SIZE);
  int *pL = (int *)(dR + PACK_ROW_SIZE);
  int *pR = (int *)(pL + PACK_ROW_SIZE);
  int offset = longWireEndRowsOffset + blockIdx.x * PACK_ROW_SIZE;
  distAtRow += offset;
  prevAtRow += offset;
  costAtRow += offset;

#pragma unroll
  for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
  {
    cL[cur] = cur == 0 ? 0.f : costAtRow[cur];
    cR[cur] = cur == 0 ? 0.f : costAtRow[PACK_ROW_SIZE - cur];
    dL[cur] = distAtRow[cur];
    dR[cur] = distAtRow[PACK_ROW_SIZE - 1 - cur];
    pL[cur] = cur;
    pR[cur] = PACK_ROW_SIZE - 1 - cur;
  }
  __syncthreads();

  for (int d = 0; (1 << d) < PACK_ROW_SIZE; d++)
  {
    int dst = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
    int src = (dst >> d << d) - 1;
    if (dL[dst] > dL[src] + cL[dst])
    {
      dL[dst] = dL[src] + cL[dst];
      pL[dst] = pL[src];
    }
    if (dR[dst] > dR[src] + cR[dst])
    {
      dR[dst] = dR[src] + cR[dst];
      pR[dst] = pR[src];
    }
    cL[dst] += cL[src];
    cR[dst] += cR[src];
    __syncthreads();
  }

#pragma unroll
  for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
  {
    realT val = dL[cur];
    int p = pL[cur];
    if (dL[cur] > dR[PACK_ROW_SIZE - 1 - cur])
    {
      val = dR[PACK_ROW_SIZE - 1 - cur];
      p = pR[PACK_ROW_SIZE - 1 - cur];
    }
    if (val < distAtRow[cur] && cur != p)
    {
      distAtRow[cur] = val;
      prevAtRow[cur] = offset + p;
    }
  }
  __syncthreads();
}

__global__ static void sweepWireGlobal(realT *distAtRow, int *prevAtRow, char *workplace,
                                       const realT *costAtRow, const int *longWireOffsets, const int *longWireLengths)
{
  int offset = longWireOffsets[blockIdx.x];
  int length = longWireLengths[blockIdx.x];
  distAtRow += offset;
  prevAtRow += offset;
  costAtRow += offset;
  workplace += offset * (4 * sizeof(realT) + 2 * sizeof(int));
  realT *cL = (realT *)workplace;
  realT *cR = (realT *)(cL + length);
  realT *dL = (realT *)(cR + length);
  realT *dR = (realT *)(dL + length);
  int *pL = (int *)(dR + length);
  int *pR = (int *)(pL + length);

  for (int cur = threadIdx.x; cur < length; cur += blockDim.x)
  {
    cL[cur] = cur == 0 ? 0.f : costAtRow[cur];
    cR[cur] = cur == 0 ? 0.f : costAtRow[length - cur];
    dL[cur] = distAtRow[cur];
    dR[cur] = distAtRow[length - 1 - cur];
    pL[cur] = cur;
    pR[cur] = length - 1 - cur;
  }
  __syncthreads();

  for (int d = 0; (1 << d) < length; d++)
  {
    for (int cur = threadIdx.x; cur < length / 2; cur += blockDim.x)
    {
      int dst = (cur >> d << (d + 1) | (1 << d)) | (cur & ((1 << d) - 1));
      int src = (dst >> d << d) - 1;
      if (dst < length)
      {
        if (dL[dst] > dL[src] + cL[dst])
        {
          dL[dst] = dL[src] + cL[dst];
          pL[dst] = pL[src];
        }
        if (dR[dst] > dR[src] + cR[dst])
        {
          dR[dst] = dR[src] + cR[dst];
          pR[dst] = pR[src];
        }
        cL[dst] += cL[src];
        cR[dst] += cR[src];
      }
    }
    __syncthreads();
  }

  for (int cur = threadIdx.x; cur < length; cur += blockDim.x)
  {
    realT val = dL[cur];
    int p = pL[cur];
    if (dL[cur] > dR[length - 1 - cur])
    {
      val = dR[length - 1 - cur];
      p = pR[length - 1 - cur];
    }
    if (val < distAtRow[cur] && cur != p)
    {
      distAtRow[cur] = val;
      prevAtRow[cur] = offset + p;
    }
  }
}

__global__ static void sweepVia(realT *distAtRow, int *prevAtRow, const realT *costAtViaseg, const int *posAtViaseg, int numViasegs)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numViasegs)
    return;
  costAtViaseg += tid * VIA_SEG_SIZE;
  posAtViaseg += tid * VIA_SEG_SIZE;
  int p[VIA_SEG_SIZE];
  realT d[VIA_SEG_SIZE], v[VIA_SEG_SIZE];
  for (int z = 0; z < VIA_SEG_SIZE; z++)
  {
    p[z] = z;
    d[z] = posAtViaseg[z] >= 0 ? distAtRow[posAtViaseg[z]] : INFINITY_DISTANCE;
    v[z] = costAtViaseg[z];
  }
  for (int z = 1; z < VIA_SEG_SIZE; z++)
  {
    if (d[z] > d[z - 1] + v[z])
    {
      d[z] = d[z - 1] + v[z];
      p[z] = p[z - 1];
    }
    if (d[VIA_SEG_SIZE - 1 - z] > d[VIA_SEG_SIZE - z] + v[VIA_SEG_SIZE - z])
    {
      d[VIA_SEG_SIZE - 1 - z] = d[VIA_SEG_SIZE - z] + v[VIA_SEG_SIZE - z];
      p[VIA_SEG_SIZE - 1 - z] = p[VIA_SEG_SIZE - z];
    }
  }
  for (int z = 0; z < VIA_SEG_SIZE; z++)
  {
    if (p[z] != z)
    {
      distAtRow[posAtViaseg[z]] = d[z];
      prevAtRow[posAtViaseg[z]] = posAtViaseg[p[z]];
    }
  }
}

__global__ static void pinIndicesToPositions(int *pinPositions, const int *pinIndices, const int *idxPosMap, int numPins)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numPins)
    pinPositions[i] = idxPosMap[pinIndices[i]];
}

__global__ static void cleanDistPrev(realT *distAtRow, int *prevAtRow, const int *markAtRow, int numRows)
{
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= numRows * PACK_ROW_SIZE)
    return;
  prevAtRow[pos] = pos;
  distAtRow[pos] = markAtRow[pos] ? 0.f : INFINITY_DISTANCE;
}

__global__ static void setRootPin(int *markAtRow, int *isRoutedPin, const int *pinPositions, int numPins)
{
  for (int i = 1; i < numPins; i++)
    isRoutedPin[i] = 0;
  isRoutedPin[0] = 1;
  markAtRow[pinPositions[0]] = 1;
}

// 回溯一条路径
__global__ static void tracePath(int *markAtRow, int *isRoutedPin, int *routes, const realT *distAtRow, const int *prevAtRow,
                                 const int *idxAtRow, const int *locAtRow, const int *posAtViaseg, const int *pinPositions, int numPins)
{
  realT minDist = INFINITY_DISTANCE;
  int pinId = -1, pinPos = -1;
  // 寻找距离最短的未被连线的pin
  for (int i = 0; i < numPins; i++)
  {
    if (!isRoutedPin[i])
    {
      int p = pinPositions[i];
      if (distAtRow[p] < minDist)
      {
        minDist = distAtRow[p];
        pinPos = p;
        pinId = i;
      }
    }
  }
  if (pinId == -1)
  {
    printf("GuideGamer: trace path failed\n");
    return;
  }
  isRoutedPin[pinId] = 1;

  int currPos = pinPos;
  int currLoc = locAtRow[currPos]; // currPos对于的gcell在Viasegs的位置
  while (!markAtRow[currPos])
  {
    int prevPos = prevAtRow[currPos];
    int prevLoc = locAtRow[prevPos];
    // fill routes
    routes[++routes[0]] = min(idxAtRow[currPos], idxAtRow[prevPos]);
    routes[++routes[0]] = max(idxAtRow[currPos], idxAtRow[prevPos]);
    // mark prev and dist
    if (currLoc >= 0 && prevLoc >= 0 && (currLoc / VIA_SEG_SIZE) == (prevLoc / VIA_SEG_SIZE)) // vias
    {
      int offset = currLoc / VIA_SEG_SIZE * VIA_SEG_SIZE;
      int currI = currLoc % VIA_SEG_SIZE;
      int prevI = prevLoc % VIA_SEG_SIZE;
      int startI = min(currI, prevI);
      int endI = max(currI, prevI);
      for (int i = startI; i <= endI; i++)
      {
        if (i != prevI)
          markAtRow[posAtViaseg[offset + i]] = 1;
      }
    }
    else // wires
    {
      int startPos = min(currPos, prevPos);
      int endPos = max(currPos, prevPos);
      for (int pos = startPos; pos <= endPos; pos++)
      {
        if (pos != prevPos)
          markAtRow[pos] = 1;
      }
    }
    currPos = prevPos;
    currLoc = prevLoc;
  }
}

// -------------------------
// GuidedGamer
// -------------------------

GuidedGamer::GuidedGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins)
    : DIRECTION(DIRECTION), N(N), X(X), Y(Y), LAYER(LAYER), numPins(0), maxNumPins(maxNumPins),
      numWires(0), maxNumWires(0), numRows(0), maxNumRows(0),
      numLongWires(0), maxNumLongWires(0), numWorkplace(0), maxNumWorkplace(0),
      numViasegs(0), maxNumViasegs(0), longWireEndRowsOffset(0)
{
  devRoutes = cuda_make_shared<int[]>(maxNumPins * MAX_ROUTE_LEN_PER_PIN);
  devIsRoutedPin = cuda_make_unique<int[]>(maxNumPins);
  devPinIndices = cuda_make_unique<int[]>(maxNumPins);
  devPinPositions = cuda_make_unique<int[]>(maxNumPins);
  devIdxPosMap = cuda_make_unique<int[]>(LAYER * N * N);
}

bool GuidedGamer::getIsRouted() const
{
  std::vector<int> isRoutePin(numPins);
  checkCudaErrors(cudaMemcpy(isRoutePin.data(), devIsRoutedPin.get(), numPins * sizeof(int), cudaMemcpyDeviceToHost));
  return std::reduce(isRoutePin.begin(), isRoutePin.end(), 1, [](int x, int y)
                     { return x & y; });
}

void GuidedGamer::setGuide(const int *routes, int scaleX, int scaleY, int coarseN)
{
  std::vector<std::pair<int3, int3>> wires;
  std::vector<std::pair<int3, int3>> viasegs;
  // collect wire segments and via segments
  for (int i = 0; i < routes[0]; i += 2)
  {
    auto [startX, startY, startZ] = idxToXYZ(routes[1 + i], DIRECTION, coarseN);
    auto [endX, endY, endZ] = idxToXYZ(routes[2 + i], DIRECTION, coarseN);
    if (startZ == endZ) // coarse wires
    {
      if ((startZ & 1) ^ DIRECTION) // Y-X
        for (int y = startY * scaleY; y < std::min((startY + 1) * scaleY, Y); y++)
          wires.emplace_back(make_int3(startX * scaleX, y, startZ), make_int3(std::min((endX + 1) * scaleX, X) - 1, y, startZ));
      else // X-Y
        for (int x = startX * scaleX; x < std::min((startX + 1) * scaleX, X); x++)
          wires.emplace_back(make_int3(x, startY * scaleY, startZ), make_int3(x, std::min((endY + 1) * scaleY, Y) - 1, startZ));
    }
    else // coarse vias
    {
      // wire segments within coarse vias
      for (int z = startZ; z <= endZ; z++)
      {
        if ((z & 1) ^ DIRECTION) // Y-X
          for (int y = startY * scaleY; y < std::min((startY + 1) * scaleY, Y); y++)
            wires.emplace_back(make_int3(startX * scaleX, y, z), make_int3(std::min((endX + 1) * scaleX, X) - 1, y, z));
        else // X-Y
          for (int x = startX * scaleX; x < std::min((startX + 1) * scaleX, X); x++)
            wires.emplace_back(make_int3(x, startY * scaleY, z), make_int3(x, std::min((endY + 1) * scaleY, Y) - 1, z));
      }
      // via segments
      for (int x = startX * scaleX; x < std::min((startX + 1) * scaleX, X); x++)
        for (int y = startY * scaleY; y < std::min((startY + 1) * scaleY, Y); y++)
          viasegs.emplace_back(make_int3(x, y, startZ), make_int3(x, y, endZ));
    }
  }
  // sort wire segments and via segments
  auto compareWire = [&](const std::pair<int3, int3> &left, const std::pair<int3, int3> &right)
  {
    const int3 &p = left.first;
    const int3 &q = right.first;
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
  auto compareViaseg = [&](const std::pair<int3, int3> &left, const std::pair<int3, int3> &right)
  {
    const int3 &p = left.first;
    const int3 &q = right.first;
    return (p.x < q.x) || (p.x == q.x && p.y < q.y) || (p.x == q.x && p.y == q.y && p.z < q.z);
  };
  std::sort(wires.begin(), wires.end(), compareWire);
  std::sort(viasegs.begin(), viasegs.end(), compareViaseg);
  // merge wire segments as far as possible
  auto merge = [&](std::vector<std::pair<int3, int3>> &segs,
                   std::function<bool(std::pair<int3, int3> &, std::pair<int3, int3> &)> doMerge)
  {
    auto first = segs.begin();
    auto last = segs.end();
    if (first == last)
      return last;
    auto result = first;
    while (++first != last)
      if (!doMerge(*result, *first) && ++result != first)
        *result = std::move(*first);
    return ++result;
  };
  auto doMergeWire = [&](std::pair<int3, int3> &dst, std::pair<int3, int3> &src)
  {
    if (dst.first.z == src.first.z)
    {
      if ((dst.first.z & 1) ^ DIRECTION)
      {
        if (dst.first.y == src.first.y && dst.second.x >= src.first.x)
        {
          dst.second.x = std::max(dst.second.x, src.second.x);
          return true;
        }
      }
      else
      {
        if (dst.first.x == src.first.x && dst.second.y >= src.first.y)
        {
          dst.second.y = std::max(dst.second.y, src.second.y);
          return true;
        }
      }
    }
    return false;
  };
  auto doMergeViaseg = [&](std::pair<int3, int3> &dst, std::pair<int3, int3> &src)
  {
    if (dst.first.x == src.first.x && dst.first.y == src.first.y && dst.second.z >= src.first.z)
    {
      dst.second.z = std::max(dst.second.z, src.second.z);
      return true;
    }
    return false;
  };
  wires.erase(merge(wires, doMergeWire), wires.end());
  viasegs.erase(merge(viasegs, doMergeViaseg), viasegs.end());

  auto getWireLength = [&](const std::pair<int3, int3> &wire)
  {
    if (wire.first.x == wire.second.x)
      return wire.second.y - wire.first.y + 1;
    else
      return wire.second.x - wire.first.x + 1;
  };
  // pack plan for wires
  std::vector<int3> wirePackPlan;
  wirePackPlan.reserve(wires.size());
  int rowsOffset = 0;
  auto longWireEndIt = std::partition(wires.begin(), wires.end(), [&](const std::pair<int3, int3> &wire)
                                      { return getWireLength(wire) > PACK_ROW_SIZE; });
  // long wire
  std::vector<int> longWireOffsets;
  std::vector<int> longWireLengths;
  for (auto it = wires.begin(); it != longWireEndIt; it++)
  {
    const auto &[p, q] = *it;
    int startIdx = xyzToIdx(p.x, p.y, p.z, DIRECTION, N);
    int endIdx = xyzToIdx(q.x, q.y, q.z, DIRECTION, N);
    if (rowsOffset % PACK_ROW_SIZE != 0)
      rowsOffset += PACK_ROW_SIZE - (rowsOffset % PACK_ROW_SIZE);
    wirePackPlan.push_back(make_int3(rowsOffset, startIdx, endIdx));
    longWireOffsets.push_back(rowsOffset);
    longWireLengths.push_back(endIdx - startIdx + 1);
    rowsOffset += (endIdx - startIdx + 1 + PACK_ROW_SIZE - 1) / PACK_ROW_SIZE * PACK_ROW_SIZE;
  }
  numLongWires = static_cast<int>(longWireOffsets.size());
  numWorkplace = rowsOffset;
  longWireEndRowsOffset = rowsOffset;
  // short wire
  for (auto it = longWireEndIt; it != wires.end(); it++)
  {
    const auto &[p, q] = *it;
    int startIdx = xyzToIdx(p.x, p.y, p.z, DIRECTION, N);
    int endIdx = xyzToIdx(q.x, q.y, q.z, DIRECTION, N);
    if (PACK_ROW_SIZE - (rowsOffset % PACK_ROW_SIZE) < endIdx - startIdx + 1) // make a new row if necessary
      rowsOffset += PACK_ROW_SIZE - (rowsOffset % PACK_ROW_SIZE);
    wirePackPlan.push_back(make_int3(rowsOffset, startIdx, endIdx));
    rowsOffset += (endIdx - startIdx + 1);
  }
  numRows = (rowsOffset + PACK_ROW_SIZE - 1) / PACK_ROW_SIZE;
  numWires = static_cast<int>(wires.size());
  // pack plan for via segments
  std::vector<int3> viasegPackPlan; // { <offset, startIdx, endIdx> }
  viasegPackPlan.reserve(viasegs.size());
  int viasegsOffset = 0;
  for (const auto &[p, q] : viasegs)
  {
    viasegPackPlan.push_back(make_int3(viasegsOffset, xyzToIdx(p.x, p.y, p.z, DIRECTION, N), xyzToIdx(q.x, q.y, q.z, DIRECTION, N)));
    viasegsOffset += VIA_SEG_SIZE;
  }
  numViasegs = static_cast<int>(viasegs.size());
  // reserve memory
  reserve(numWires, numRows, numLongWires, numWorkplace, numViasegs);

  printf("numLongWires: %d\n", numLongWires);
  printf("numWorkplace: %d\n", numWorkplace);
  printf("longWireEndRowsOffset: %d\n", longWireEndRowsOffset);
  printf("numRows: %d\n", numRows);
  printf("numWires: %d\n", numWires);
  printf("numViasegs: %d\n", numViasegs);

  printf("wires:\n");
  for (int i = 0; i < wires.size(); i++)
  {
    const auto &[p, q] = wires[i];
    const auto &[offset, startIdx, endIdx] = wirePackPlan[i];
    printf("offset: %d, len: %d : (%d, %d, %d) -> (%d, %d, %d)\n", offset, endIdx - startIdx + 1, p.x, p.y, p.z, q.x, q.y, q.z);
  }
  printf("viasegs:\n");
  for (int i = 0; i < viasegs.size(); i++)
  {
    const auto &[p, q] = viasegs[i];
    const auto &[offset, startIdx, endIdx] = viasegPackPlan[i];
    printf("offset: %d: (%d, %d, %d) -> (%d, %d, %d)\n", offset, p.x, p.y, p.z, q.x, q.y, q.z);
  }
  printf("longwires:\n");
  for (int i = 0; i < longWireLengths.size(); i++)
  {
    printf("offset: %d, length: %d\n", longWireOffsets[i], longWireLengths[i]);
  }
  printf("numWorkplace: %d\n", numWorkplace);

  // pack row and viaseg
  checkCudaErrors(cudaMemcpy(devWirePackPlan.get(), wirePackPlan.data(), wirePackPlan.size() * sizeof(int3), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devViasegPackPlan.get(), viasegPackPlan.data(), viasegPackPlan.size() * sizeof(int3), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devLongWireLengths.get(), longWireLengths.data(), longWireLengths.size() * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devLongWireOffsets.get(), longWireOffsets.data(), longWireOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(devIdxPosMap.get(), -1, LAYER * N * N * sizeof(int)));
  checkCudaErrors(cudaMemset(devIdxAtRow.get(), -1, numRows * PACK_ROW_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemset(devLocAtRow.get(), -1, numRows * PACK_ROW_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemset(devPosAtViaseg.get(), -1, numViasegs * VIA_SEG_SIZE * sizeof(int)));
  clearCostAtRow<<<(numRows * PACK_ROW_SIZE + 1023) / 1024, 1024>>>(devCostAtRow.get(), numRows);
  clearCostAtViaseg<<<(numViasegs * VIA_SEG_SIZE + 1023) / 1024, 1024>>>(devCostAtViaseg.get(), numViasegs);
  packWireToRows<<<(numWires + 1023) / 1024, 1024>>>(
      devIdxAtRow.get(), devCostAtRow.get(), devIdxPosMap.get(),
      devWirePackPlan.get(), devWireCost.get(), numWires, DIRECTION, N);
  packViaToSegs<<<(numViasegs + 1023) / 1024, 1024>>>(
      devCostAtViaseg.get(), devPosAtViaseg.get(), devLocAtRow.get(),
      devViasegPackPlan.get(), devViaCost.get(), devIdxPosMap.get(), numViasegs, DIRECTION, N);
  checkCudaErrors(cudaDeviceSynchronize());
}

void GuidedGamer::reserve(int nWires, int nRows, int nLongWires, int nWorkplace, int nViasegs)
{
  if (maxNumWires < nWires)
  {
    utils::log() << "gamer info: reserve for wires\n";
    maxNumWires = std::max(nWires, maxNumWires * 2);
    devWirePackPlan = cuda_make_unique<int3[]>(maxNumWires);
  }
  if (maxNumRows < nRows)
  {
    utils::log() << "gamer info: reserve for rows\n";
    maxNumRows = std::max(nRows, maxNumRows * 2);
    devIdxAtRow = cuda_make_unique<int[]>(maxNumRows * PACK_ROW_SIZE);
    devLocAtRow = cuda_make_unique<int[]>(maxNumRows * PACK_ROW_SIZE);
    devCostAtRow = cuda_make_unique<realT[]>(maxNumRows * PACK_ROW_SIZE);
    devDistAtRow = cuda_make_unique<realT[]>(maxNumRows * PACK_ROW_SIZE);
    devPrevAtRow = cuda_make_unique<int[]>(maxNumRows * PACK_ROW_SIZE);
    devMarkAtRow = cuda_make_unique<int[]>(maxNumRows * PACK_ROW_SIZE);
  }
  if (maxNumLongWires < numLongWires)
  {
    utils::log() << "gamer info: reserve for longwires\n";
    maxNumLongWires = std::max(nLongWires, 2 * maxNumLongWires);
    devLongWireLengths = cuda_make_unique<int[]>(maxNumLongWires);
    devLongWireOffsets = cuda_make_unique<int[]>(maxNumLongWires);
  }
  if (maxNumWorkplace < nWorkplace)
  {
    utils::log() << "gamer info: reserve for workplace\n";
    maxNumWorkplace = std::max(nWorkplace, maxNumWorkplace * 2);
    devWorkplace = cuda_make_unique<char[]>(maxNumWorkplace * (4 * sizeof(realT) + 2 * sizeof(int)));
  }
  if (maxNumViasegs < nViasegs)
  {
    utils::log() << "gamer info: reserve for viasegs\n";
    maxNumViasegs = std::max(nViasegs, maxNumViasegs * 2);
    devViasegPackPlan = cuda_make_unique<int3[]>(maxNumViasegs);
    devPosAtViaseg = cuda_make_unique<int[]>(maxNumViasegs * VIA_SEG_SIZE);
    devCostAtViaseg = cuda_make_unique<realT[]>(maxNumViasegs * VIA_SEG_SIZE);
  }
}

void GuidedGamer::route(const std::vector<int> &pinIndices, int sweepTurns)
{
  numPins = static_cast<int>(pinIndices.size());
  checkCudaErrors(cudaMemset(devMarkAtRow.get(), 0, numRows * PACK_ROW_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemset(devRoutes.get(), 0, maxNumPins * MAX_ROUTE_LEN_PER_PIN * sizeof(int)));
  checkCudaErrors(cudaMemset(devIsRoutedPin.get(), 0, numPins * sizeof(int)));
  checkCudaErrors(cudaMemcpy(devPinIndices.get(), pinIndices.data(), numPins * sizeof(int), cudaMemcpyHostToDevice));
  pinIndicesToPositions<<<numPins, 1>>>(devPinPositions.get(), devPinIndices.get(), devIdxPosMap.get(), numPins);
  std::vector<int> pinPositions(numPins);
  checkCudaErrors(cudaMemcpy(pinPositions.data(), devPinPositions.get(), numPins * sizeof(int), cudaMemcpyDeviceToHost));
  utils::log() << "pin idx -> pin pos:\n";
  for(int i = 0; i < numPins; i++)
  {
    auto [x, y, z] = idxToXYZ(pinIndices[i], DIRECTION, N);
    int r = pinPositions[i] / PACK_ROW_SIZE;
    int c = pinPositions[i] % PACK_ROW_SIZE;
    printf("(%d, %d, %d) -> (%d, %d)\n", x, y, z, r, c);
  }
  setRootPin<<<1, 1>>>(devMarkAtRow.get(), devIsRoutedPin.get(), devPinPositions.get(), numPins);
  cleanDistPrev<<<(numRows * PACK_ROW_SIZE + 1023) / 1024, 1024>>>(devDistAtRow.get(), devPrevAtRow.get(), devMarkAtRow.get(), numRows);
  for (int iter = 1; iter < pinIndices.size(); iter++)
  {
    for (int turn = 0; turn < sweepTurns; turn++)
    {
      sweepVia<<<(numViasegs + 1023) / 1024, 1024>>>(
          devDistAtRow.get(), devPrevAtRow.get(), devCostAtViaseg.get(), devPosAtViaseg.get(), numViasegs);
      sweepWireGlobal<<<numLongWires, PACK_ROW_SIZE / 2>>>(
          devDistAtRow.get(), devPrevAtRow.get(), devWorkplace.get(), devCostAtRow.get(), devLongWireOffsets.get(), devLongWireLengths.get());
      sweepWireShared<<<numRows - (longWireEndRowsOffset / PACK_ROW_SIZE), PACK_ROW_SIZE / 2>>>(
          devDistAtRow.get(), devPrevAtRow.get(), devCostAtRow.get(), longWireEndRowsOffset);
    }
    tracePath<<<1, 1>>>(
        devMarkAtRow.get(), devIsRoutedPin.get(), devRoutes.get(), devDistAtRow.get(), devPrevAtRow.get(),
        devIdxAtRow.get(), devLocAtRow.get(), devPosAtViaseg.get(), devPinPositions.get(), numPins);
    cleanDistPrev<<<(numRows * PACK_ROW_SIZE + 1023) / 1024, 1024>>>(devDistAtRow.get(), devPrevAtRow.get(), devMarkAtRow.get(), numRows);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}
#endif