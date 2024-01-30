#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_GRID_SCALER_2D_H
#define GPU_ROUTE_GRID_SCALER_2D_H

#include "gamer_utils.cuh"

class GridScaler2D
{
public:
  GridScaler2D(int X, int Y, int scaleX, int scaleY);

  int getScaleX() const { return scaleX; }
  int getScaleY() const { return scaleY; }
  int getCoarseX() const { return coarseX; }
  int getCoarseY() const { return coarseY; }

  void setCost2D(const cuda_shared_ptr<realT[]> &cost2D) { devCost2D = cost2D; }
  const cuda_shared_ptr<realT[]> &getCoarseCost2D() const { return devCoarseCost2D; }

  void scalePin2DIndices(std::vector<int> &coarsePin2DIndices, const std::vector<int> &pin2DIndices) const;
  utils::BoxT<int> coarsenBoudingBox(const utils::BoxT<int> &box) const;
  utils::BoxT<int> finingBoundingBox(const utils::BoxT<int> &box) const;
  void getGuideFromRoutes2D(std::vector<utils::BoxT<int>> &guide2D, const int *routes2D) const;

  void scaleCost2D();
  void scaleCost2D(const utils::BoxT<int> &coarseBox);

private:
  int X, Y;
  int scaleX, scaleY;
  int coarseX, coarseY;

  cuda_shared_ptr<realT[]> devCost2D;
  cuda_shared_ptr<realT[]> devCoarseCost2D;
};

#endif
#endif