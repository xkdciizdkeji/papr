#pragma once
#include "GRNet.h"
#include "GridGraph.h"
#include "PatternRoute.h"

class GPUMazeRoute
{
public:
  GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params);
  GPUMazeRoute(const GPUMazeRoute &) = delete;
  GPUMazeRoute(GPUMazeRoute &&) = delete;
  GPUMazeRoute &operator=(const GPUMazeRoute &) = delete;
  GPUMazeRoute &operator=(GPUMazeRoute &&) = delete;
  ~GPUMazeRoute();

  void route(const std::vector<int> &netIndices, int sweepTurns);
  void commit(const std::vector<int> &netIndices);
  void getOverflowNetIndices(std::vector<int> &netIndices) const;

private:
  std::pair<std::vector<int>, std::vector<int>> batching(const std::vector<int> &netIndices) const;
  std::shared_ptr<GRTreeNode> extractGRTree(int *treeMap, const int *routes, int rootIdx) const;
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

  float unitLengthWireCost;
  float unitViaCost;
  float logisticSlope;
  float viaMultiplier;

  float *devHEdgeLengths;
  float *devVEdgeLengths;
  float *devLayerMinLengths;
  float *devUnitLengthShortCosts;

  float *devCapacity;
  float *devDemand;

  int *cpuRootIndices;
  int *cpuIsRoutedNet;

  int *cpuRoutes, *devRoutes;
  int *cpuRoutesOffset, *devRoutesOffset;

  int *cpuIsOverflowNet, *devIsOverflowNet;
  int *cpuNetIndices, *devNetIndices;

  int *cpuAllpins, *devAllpins;
  int *cpuIsRoutedPin, *devIsRoutedPin;

  float *devWireCost;
  float *devWireCostSum;
  float *devViaCost;

  float *devDist;
  int *devPrev;

  int *cpuIsIdentical, *devIsIdentical;
  int *devLastPrev;
};
