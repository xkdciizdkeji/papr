#include "GPUMazeRoute.h"
#include "../gamer/GPUMazeRouter.cuh"

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
{
  router = std::make_unique<GPUMazeRouter>(nets, graph, params);
}

GPUMazeRoute::~GPUMazeRoute()
{
  router.release();
}

void GPUMazeRoute::route(const std::vector<int> &netIndices, int sweepTurns, int margin)
{
  router->route(netIndices, sweepTurns, margin);
}

void GPUMazeRoute::commit(const std::vector<int> &netIndices)
{
  router->commit(netIndices);
}

void GPUMazeRoute::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  router->getOverflowNetIndices(netIndices);
}
