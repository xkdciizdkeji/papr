#include "GPUMazeRoute.h"
#include "../gamer/GPURouteContext.cuh"
#include "../gamer/GPUMazeRouteBasic.cuh"
#include "../gamer/GPUMazeRouteTwostep.cuh"
#include "../gamer/GPUMazeRouteTwostep3D.cuh"

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
  auto router = std::make_unique<GPUMazeRouteBasic>(context);
  // auto router = std::make_unique<GPUMazeRouteTwostep>(context);
  // auto router = std::make_unique<GPUMazeRouteTwostep3D>(context);
  log() << "gamer info. init done\n";

  log() << "gamer info. routing ...\n";
  std::vector<int> netIndices;
  context->getOverflowAndOpenNetIndices(netIndices);
  std::sort(netIndices.begin(), netIndices.end(), [&](int left, int right) {
    return nets[left].getBoundingBox().hp() < nets[right].getBoundingBox().hp();
  });
  // netIndices.resize(500);
  router->run(netIndices, 8, 10);
  // router->run(netIndices, 2, 8, 10);
  // router->run(netIndices, 5, 8, 10);
  log() << "gamer info. routing done\n";

  log() << "gamer info. apply to cpu ...\n";
  context->applyToCpu(netIndices);
  log() << "gamer info. apply to cpu done\n";
}