#include "GPUMazeRoute.h"
#include "../gamer/GPUMazeRouter.cuh"

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
{
  router = std::make_unique<GPUMazeRouter>(nets, graph, params);
}

GPUMazeRoute::~GPUMazeRoute()
{
}

void GPUMazeRoute::route(const std::vector<int> &netIndices, int sweepTurn, int margin)
{
  router->route(netIndices, sweepTurn, margin);
  // router->routeTwoStep(netIndices, 5, 8, 10);
}

void GPUMazeRoute::apply(const std::vector<int> &netIndices)
{
  router->apply(netIndices);
}

void GPUMazeRoute::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  router->getOverflowNetIndices(netIndices);
}
