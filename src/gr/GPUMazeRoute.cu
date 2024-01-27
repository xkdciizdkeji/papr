#include "GPUMazeRoute.h"
#include "../gamer/GPUMazeRouter.cuh"

GPUMazeRoute::GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params)
    : nets(nets), gridGraph(graph), parameters(params)
{
  router = std::make_unique<GPUMazeRouter>(nets, graph, params);
}

GPUMazeRoute::~GPUMazeRoute()
{
}

void GPUMazeRoute::run()
{
  log() << "gamer info. routing ...\n";
  std::vector<int> netIndices(nets.size());
  // router->getOverflowNetIndices(netIndices);
  for(int i = 0; i < nets.size(); i++)
    netIndices[i] = i;
  // router->route(netIndices, 9, 20);
  router->routeTwoStep(netIndices, 2, 5, 10);
  log() << "gamer info. routing done\n";

  log() << "gamer info. commiting gamer's result ...\n";
  router->applyToCpu(netIndices);
  log() << "gamer info. commiting done\n";
}

// void GPUMazeRoute::run()
// {
//   log() << "gamer info. routing ...\n";
//   std::vector<int> netIndices;
//   std::vector<bool> isGamerRoutedNet(nets.size(), false);
//   for(int k = 0; k <= 3; k++)
//   {
//     router->getOverflowNetIndices(netIndices);
//     utils::log() << "gamer iter " << k << ". overflow net: " << netIndices.size() << " / " << nets.size() << "\n";
//     for(auto id : netIndices)
//       isGamerRoutedNet[id] = true;
//     router->routeTwoStep(netIndices, k + 3, 2 * (k + 3) + 1, 10 * (k + 3));
//     // router->route(netIndices, 9, 20);
//     // router->routeSparse(netIndices, 9, 10);
//   }
//   log() << "gamer info. routing done\n";

//   log() << "gamer info. commiting gamer's result ...\n";
//   netIndices.clear();
//   for(int id = 0; id < isGamerRoutedNet.size(); id++)
//     if(isGamerRoutedNet[id])
//       netIndices.push_back(id);
//   router->applyToCpu(netIndices);
//   log() << "gamer info. commiting done\n";
// }

void GPUMazeRoute::getOverflowNetIndices(std::vector<int> &netIndices) const
{
  router->getOverflowNetIndices(netIndices);
}
