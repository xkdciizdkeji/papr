#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_BASIC_GAMER_2D_H
#define GPU_ROUTE_BASIC_GAMER_2D_H

#include "gamer_utils.cuh"

class BasicGamer2D
{
public:
  BasicGamer2D(int X, int Y, int maxNumPins);

  void setCost2D(const cuda_shared_ptr<realT[]> &cost2D) { devCost = cost2D; }
  const cuda_shared_ptr<int[]> &getRoutes2D() const { return devRoutes; }
  bool getIsRouted() const;

  void route(const std::vector<int> &pin2DIndices, int numTurns);
  void route(const std::vector<int> &pin2DIndices, int numTurns, const utils::BoxT<int> &box);

private:
  int X, Y;
  int numPins, maxNumPins;

  cuda_shared_ptr<int[]> devRoutes;
  cuda_unique_ptr<int[]> devIsRoutedPin;
  cuda_unique_ptr<int[]> devPinIndices;

  cuda_unique_ptr<realT[]> devCostSum;
  cuda_unique_ptr<realT[]> devDist;
  cuda_unique_ptr<int[]> devAllPrev;
  cuda_unique_ptr<int[]> devMark;

  cuda_shared_ptr<realT[]> devCost;
};

#endif
#endif