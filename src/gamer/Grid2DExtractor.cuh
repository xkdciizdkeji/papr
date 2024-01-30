#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_GRID_2D_EXTRACTOR_H
#define GPU_ROUTE_GRID_2D_EXTRACTOR_H

#include "gamer_utils.cuh"

class Grid2DExtractor
{
public:
  Grid2DExtractor(int DIRECTION, int N, int X, int Y, int LAYER);

  void setWireCost(const cuda_shared_ptr<realT[]> &wireCost) { devWireCost = wireCost; }
  const cuda_shared_ptr<realT[]> &getCost2D() { return devCost2D; }

  void extractCost2D();
  void extractCost2D(const utils::BoxT<int> &box);

  void extractPin2DIndices(std::vector<int> &pin2DIndices, const std::vector<int> &pinIndices) const;

private:
  int DIRECTION, N, X, Y, LAYER;
  
  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devCost2D;
};

#endif
#endif