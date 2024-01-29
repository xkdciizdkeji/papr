#ifdef ENABLE_CUDA
#ifndef GPU_TWOSTEP_MAZE_ROUTE_H
#define GPU_TWOSTEP_MAZE_ROUTE_H

#include "GPURouteContext.cuh"
#include "BasicGamer.cuh"
#include "BasicGamer2D.cuh"
#include "GridScaler2D.cuh"
#include "Grid2DExtractor.cuh"
#include "GuidedGamer.cuh"

class GPUMazeRouteTwostep
{
public:
  GPUMazeRouteTwostep(const std::shared_ptr<GPURouteContext> &context);
  void run(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int margin);

private:
  void getCoarsePin2DIndices(std::vector<int> &coarsePin2DIndices, const std::vector<int> &pinIndices);
  void getGuideFromCoarseRoutes2D(std::vector<std::array<int, 6>> &guide, const std::vector<int> &coarseRoutes2D);

private:
  std::shared_ptr<GPURouteContext> context;
  std::unique_ptr<Grid2DExtractor> extractor;
  std::unique_ptr<GridScaler2D> scaler;
  std::unique_ptr<BasicGamer2D> coarseGamer;
  std::unique_ptr<GuidedGamer> fineGamer;
};

#endif
#endif