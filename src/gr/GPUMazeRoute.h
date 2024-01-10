#pragma once
#ifdef ENABLE_CUDA
#include "GRNet.h"
#include "GridGraph.h"
#include "PatternRoute.h"

using realT = double;

class GPUMazeRoute
{
public:
  GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params);
  GPUMazeRoute(const GPUMazeRoute &) = delete;
  GPUMazeRoute(GPUMazeRoute &&) = delete;
  GPUMazeRoute &operator=(const GPUMazeRoute &) = delete;
  GPUMazeRoute &operator=(GPUMazeRoute &&) = delete;
  ~GPUMazeRoute();

  void route(const std::vector<int> &netIndices, int sweepTurns, int margin);
  void commit(const std::vector<int> &netIndices);
  void getOverflowNetIndices(std::vector<int> &netIndices) const;

private:
  std::pair<std::vector<int>, std::vector<int>> batching(const std::vector<int> &netIndices) const;
  std::shared_ptr<GRTreeNode> extractGRTree(const int *routes, int rootIdx) const;
  std::vector<int> selectAccessPoints(const GRNet &net) const;

private:
  const Parameters &parameters;
  GridGraph &gridGraph;
  std::vector<GRNet> &nets;

  int DIRECTION;
  int X, Y, LAYER;
  int N;
  int NUMNET;
  int ALLPIN_STRIDE;

  realT unitLengthWireCost;
  realT unitViaCost;
  realT logisticSlope;
  realT viaMultiplier;

  realT *devHEdgeLengths;
  realT *devVEdgeLengths;
  realT *devLayerMinLengths;
  realT *devUnitLengthShortCosts;

  realT *devCapacity;
  realT *devDemand;

  int *cpuRootIndices;
  int *cpuIsRoutedNet;

  int *cpuRoutes, *devRoutes;
  int *cpuRoutesOffset, *devRoutesOffset;

  int *cpuIsOverflowNet, *devIsOverflowNet;
  int *cpuNetIndices, *devNetIndices;

  int *cpuAllpins, *devAllpins;
  int *cpuIsRoutedPin, *devIsRoutedPin;

  realT *devWireCost;
  realT *devWireCostSum;
  realT *devViaCost;

  realT *devDist;
  int *devPrev;
};
#endif