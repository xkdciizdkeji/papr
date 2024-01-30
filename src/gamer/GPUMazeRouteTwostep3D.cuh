#ifdef ENABLE_CUDA
#ifndef GPU_MAZE_ROUTE_TWOSTEP_3D_H
#define GPU_MAZE_ROUTE_TWOSTEP_3D_H

#include "GPURouteContext.cuh"
#include "GridScaler.cuh"
#include "BasicGamer.cuh"
#include "GuidedGamer.cuh"

class GPUMazeRouteTwostep3D
{
public:
  GPUMazeRouteTwostep3D(const std::shared_ptr<GPURouteContext> &context);
  void run(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int margin);

private:
  void getCoarsePinIndices(std::vector<int> &coarsePinIndices, const std::vector<int> &pinIndices) const;
  void getGuide(std::vector<std::array<int, 6>> &guide, const std::vector<int> &coarseRoutes) const;
  void getGuide(std::vector<std::array<int, 6>> &guide, const std::vector<int> &coarsePinIndices, const std::vector<int> &coarseRoutes) const;

private:
  std::shared_ptr<GPURouteContext> context;
  std::unique_ptr<GridScaler> scaler;
  std::unique_ptr<BasicGamer> coarseGamer;
  std::unique_ptr<GuidedGamer> fineGamer;
};

#endif
#endif