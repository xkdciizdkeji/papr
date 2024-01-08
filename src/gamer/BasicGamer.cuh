#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_BASIC_GAMER_H
#define GPU_ROUTE_BASIC_GAMER_H

#include "gamer_utils.cuh"

class BasicGamer
{
public:
  BasicGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins);

  void setWireCostMap(const cuda_shared_ptr<const realT[]> &wireCost) { devWireCost = wireCost; }
  void setViaCostMap(const cuda_shared_ptr<const realT[]> &viaCost) { devViaCost = viaCost; }
  cuda_shared_ptr<const int[]> getRoutes() const { return devRoutes; }
  bool getIsRouted() const;

  void route(const std::vector<int> &pinIndices, int sweepTurns);
  void route(const std::vector<int> &pinIndices, int sweepTurns, const utils::BoxT<int> &box);

private:
  int DIRECTION, N, X, Y, LAYER;
  int numPins, maxNumPins;

  cuda_shared_ptr<int[]> devRoutes;
  cuda_unique_ptr<int[]> devIsRoutedPin;
  cuda_unique_ptr<int[]> devPinIndices;

  cuda_unique_ptr<realT[]> devWireCostSum;
  cuda_unique_ptr<realT[]> devDist;
  cuda_unique_ptr<int[]> devPrev;

  cuda_shared_ptr<const realT[]> devWireCost;
  cuda_shared_ptr<const realT[]> devViaCost;
};

#endif
#endif