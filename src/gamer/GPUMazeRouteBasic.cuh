#ifdef ENABLE_CUDA
#ifndef GPU_MAZE_ROUTE_H
#define GPU_MAZE_ROUTE_H

#include "GPURouteContext.cuh"
#include "BasicGamer.cuh"

class GPUMazeRouteBasic
{
public:
  GPUMazeRouteBasic(const std::shared_ptr<GPURouteContext> &context);
  void run(const std::vector<int> &netIndices, int numTurns, int margin);

private:
  std::shared_ptr<GPURouteContext> context;
  std::unique_ptr<BasicGamer> gamer;
};

#endif
#endif