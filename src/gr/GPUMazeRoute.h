#pragma once
#include "GRNet.h"
#include "GridGraph.h"

class GPUMazeRoute
{
public:
  GPUMazeRoute(std::vector<GRNet> &nets, GridGraph &gridGraph, const Parameters &parameters);
  ~GPUMazeRoute();

  void run();

private:
  const Parameters &parameters;
  GridGraph &gridGraph;
  std::vector<GRNet> &nets;
};
