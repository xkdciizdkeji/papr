#include "GPUMazeRoute.h"
#include "../gamer/GPUMazeRouter.cuh"

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
{
  router = std::make_unique<GPUMazeRouter>(nets, graph, params);
}

GPUMazeRoute::~GPUMazeRoute()
{
}

void GPUMazeRoute::run()
{
  log() << "gamer info. routing ...\n";
  std::vector<int> netIndices;
  router->getOverflowNetIndices(netIndices);
  // router->route(netIndices, 9, 20);
  router->routeTwoStep(netIndices, 4, 9, 20);
  log() << "gamer info. routing done\n";

  log() << "gamer info. commiting gamer's result ...\n";
  router->applyToCpu(netIndices);
  log() << "gamer info. commiting done\n";
}

void GPUMazeRoute::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  router->getOverflowNetIndices(netIndices);
}