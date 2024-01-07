#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_GRID_SCALER_H
#define GPU_ROUTE_GRID_SCALER_H

#include "gamer_utils.cuh"

class GridScaler
{
public:
  GridScaler(int DIRECTION, int N, int X, int Y, int LAYER, int scaleX, int scaleY);

  int getScaleX() const { return scaleX; }
  int getScaleY() const { return scaleY; }
  int getCoarseN() const { return coarseN; }
  int getCoarseX() const { return coarseX; }
  int getCoarseY() const { return coarseY; }

  void setWireCostMap(const cuda_shared_ptr<const realT[]> &wireCost) { devWireCost = wireCost; }
  void setViaCostMap(const cuda_shared_ptr<const realT[]> &viaCost) { devViaCost = viaCost; }
  cuda_shared_ptr<const realT[]> getCoarseWireCost() const { return devCoarseWireCost; }
  cuda_shared_ptr<const realT[]> getCoarseViaCost() const { return devCoarseViaCost; }

  std::vector<int> calculateCoarsePinIndices(const std::vector<int> &pinIndices);
  utils::BoxT<int> calculateCoarseBoudingBox(const utils::BoxT<int> &box);

  void scale();
  void scale(const utils::BoxT<int> &coarseBox);

private:
  int DIRECTION, N, X, Y, LAYER;
  int scaleX, scaleY;
  int coarseN, coarseX, coarseY;

  cuda_shared_ptr<const realT[]> devWireCost;
  cuda_shared_ptr<const realT[]> devViaCost;

  cuda_shared_ptr<realT[]> devCoarseWireCost;
  cuda_shared_ptr<realT[]> devCoarseViaCost;
};

#endif
#endif