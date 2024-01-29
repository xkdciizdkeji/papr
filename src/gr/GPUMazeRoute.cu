#include "GPUMazeRoute.h"
#include "../gamer/GPURouteContext.cuh"
#include "../gamer/GPUMazeRouteBasic.cuh"
#include "../gamer/GPUMazeRouteTwostep.cuh"

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
    : nets(nets), gridGraph(graph), parameters(params)
{
}

GPUMazeRoute::~GPUMazeRoute()
{
}

void GPUMazeRoute::run()
{
  log() << "gamer info. init ...\n";
  auto context = std::make_shared<GPURouteContext>(nets, gridGraph, parameters);
  // auto router = std::make_unique<GPUMazeRouteBasic>(context);
  auto router = std::make_unique<GPUMazeRouteTwostep>(context);
  log() << "gamer info. init done\n";

  log() << "gamer info. routing ...\n";
  std::vector<int> netIndices{ 17887 };
  context->getOverflowAndOpenIndices(netIndices);
  router->run(netIndices, 2, 5, 10);
  log() << "gamer info. routing done\n";

  log() << "gamer info. apply to cpu ...\n";
  context->applyToCpu(netIndices);
  log() << "gamer info. apply to cpu done\n";
}
