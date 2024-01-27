#pragma once
#include "GRNet.h"
#include "GridGraph.h"

class GPUMazeRouter;

class GPUMazeRoute
{
public:
  GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &graph, const Parameters &params);
  ~GPUMazeRoute();

  void run();
  void getOverflowNetIndices(std::vector<int> &netIndices) const;
private:
  const Parameters &parameters;
  const GridGraph &gridGraph;
  const std::vector<GRNet> &nets;
  std::unique_ptr<GPUMazeRouter> router;
};
