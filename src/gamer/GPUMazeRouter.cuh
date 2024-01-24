#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_MAZE_ROUTER_H
#define GPU_ROUTE_MAZE_ROUTER_H

#include "../gr/GRNet.h"
#include "../gr/GridGraph.h"
#include "gamer_utils.cuh"
#include "BasicGamer.cuh"
#include "Grid2DExtractor.cuh"
#include "GridScaler2D.cuh"
#include "BasicGamer2D.cuh"
#include "GuidedGamer.cuh"

class GPUMazeRouter
{
public:
  GPUMazeRouter(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params);

  // void route(const std::vector<int> &netIndices, int numTurns, int margin);
  void routeTwoStep(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int coarseMargin);
  void applyToCpu(const std::vector<int> &netIndices);
  void getOverflowNetIndices(std::vector<int> &netIndices) const;

private:
  void selectAccessPoints(const GRNet &net, std::vector<int> &pinIndices) const;
  utils::BoxT<int> getPinIndicesBoundingBox(const std::vector<int> &pinIndices, int DIRECTION, int N) const;
  std::shared_ptr<GRTreeNode> extractGRTree(const int *routes, int rootIdx) const;

private:
  const Parameters &parameters;
  GridGraph &gridGraph;
  std::vector<GRNet> &nets;

  int DIRECTION;
  int X, Y, LAYER;
  int N;
  int maxNumPins;

  realT unitLengthWireCost;
  realT unitViaCost;
  realT logisticSlope;
  realT viaMultiplier;

  cuda_unique_ptr<realT[]> devHEdgeLengths;
  cuda_unique_ptr<realT[]> devVEdgeLengths;
  cuda_unique_ptr<realT[]> devLayerMinLengths;
  cuda_unique_ptr<realT[]> devUnitLengthShortCosts;

  cuda_unique_ptr<realT[]> devCapacity;
  cuda_unique_ptr<realT[]> devDemand;
  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devViaCost;

  std::vector<int> rootIndices;
  std::vector<int> isRoutedNet;
  std::vector<int> isOverflowNet;
  cuda_unique_ptr<int[]> devIsOverflowNet;
  cuda_unique_ptr<int[]> devNetIndices;

  std::vector<int> allRoutes;
  cuda_unique_ptr<int[]> devAllRoutes;
  std::vector<int> allRoutesOffset;
  cuda_unique_ptr<int[]> devAllRoutesOffset;

  // one-step routing
  // std::unique_ptr<BasicGamer> gamer;
  // two-step routing
  std::unique_ptr<Grid2DExtractor> extractor2D;
  std::unique_ptr<GridScaler2D> scaler2D;
  std::unique_ptr<BasicGamer2D> coarseGamer2D;
  std::unique_ptr<GuidedGamer> fineGamer;
};

#endif
#endif