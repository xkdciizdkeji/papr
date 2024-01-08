#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_MAZE_ROUTER_H
#define GPU_ROUTE_MAZE_ROUTER_H

#include "../gr/GRNet.h"
#include "../gr/GridGraph.h"
#include "gamer_utils.cuh"
#include "BasicGamer.cuh"
#include "GuidedGamer.cuh"
#include "GridScaler.cuh"

class GPUMazeRouter
{
public:
  GPUMazeRouter(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params);

  void route(const std::vector<int> &netIndices, int sweepTurns, int margin);
  void commit(const std::vector<int> &netIndices);
  void getOverflowNetIndices(std::vector<int> &netIndices) const;

private:
  std::vector<int> selectAccessPoints(const GRNet &net) const;

private:
  const Parameters &parameters;
  GridGraph &gridGraph;
  std::vector<GRNet> &nets;

  int DIRECTION;
  int X, Y, LAYER;
  int N;
  int numNets;
  int maxNumPins;
  int scaleX, scaleY;

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

  std::unique_ptr<BasicGamer> basicGamer;
  std::unique_ptr<GuidedGamer> guidedGamer;
  std::unique_ptr<GridScaler> gridScaler;
};

#endif
#endif