#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_BASIC_GAMER_H
#define GPU_ROUTE_BASIC_GAMER_H

#include "gamer_utils.cuh"

class BasicGamer
{
public:
  BasicGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins);

  void setWireCost(const cuda_shared_ptr<realT[]> &cost) { devWireCost = cost; }
  void setNonStackViaCost(const cuda_shared_ptr<realT[]> &cost) { devNonStackViaCost = cost; }
  void setUnitViaCost(realT cost) { unitViaCost = cost; }
  const cuda_shared_ptr<int[]> &getRoutes() const { return devRoutes; }
  bool getIsRouted() const;

  void route(const std::vector<int> &pinIndices, int numTurns);
  void route(const std::vector<int> &pinIndices, int numTurns, const utils::BoxT<int> &box);

private:
  int DIRECTION, N, X, Y, LAYER;
  int numPins, maxNumPins;

  cuda_shared_ptr<int[]> devRoutes;
  cuda_unique_ptr<int[]> devIsRoutedPin;
  cuda_unique_ptr<int[]> devPinIndices;

  cuda_unique_ptr<realT[]> devWireCostSum;
  cuda_unique_ptr<realT[]> devDist;
  cuda_unique_ptr<int[]> devAllPrev;
  cuda_unique_ptr<int[]> devMark;

  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devNonStackViaCost;
  realT unitViaCost;
};

#endif
#endif