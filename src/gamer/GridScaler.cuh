#ifdef ENABLE_CUDA
#ifndef GRID_SCALER_H
#define GRID_SCALER_H

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

  void setWireCost(const cuda_shared_ptr<realT[]> &wireCost) { devWireCost = wireCost; }
  void setViaCost(const cuda_shared_ptr<realT[]> &viaCost) { devViaCost = viaCost; }
  const cuda_shared_ptr<realT[]> &getCoarseWireCost() const { return devCoarseWireCost; }
  const cuda_shared_ptr<realT[]> &getCoarseViaCost() const { return devCoarseViaCost; }

  void scale();
  void scale(const utils::BoxT<int> &coarseBox);

private:
  int DIRECTION, N, X, Y, LAYER;
  int scaleX, scaleY;
  int coarseN, coarseX, coarseY;

  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devViaCost;
  cuda_shared_ptr<realT[]> devCoarseWireCost;
  cuda_shared_ptr<realT[]> devCoarseViaCost;
};
#endif
#endif