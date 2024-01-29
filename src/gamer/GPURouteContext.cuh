#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_BASE_H
#define GPU_ROUTE_BASE_H

#include "../gr/GRNet.h"
#include "../gr/GridGraph.h"
#include "gamer_utils.cuh"
#include <array>

class GPURouteContext
{
public:
  GPURouteContext(std::vector<GRNet> &nets, GridGraph &gridGraph, const Parameters &parameters);

  int getDIRECTION() const { return DIRECTION; }
  int getX() const { return X; }
  int getY() const { return Y; }
  int getN() const { return N; }
  int getLAYER() const { return LAYER; }
  int getMaxNumPins() const { return maxNumPins; }

  const cuda_shared_ptr<realT[]> &getWireCost() const { return devWireCost; }
  const cuda_shared_ptr<realT[]> &getNonStackViaCost() const { return devNonStackViaCost; }
  realT getUnitViaCost() const { return unitViaCost; }

  const std::vector<int> &getPinIndices(int netId) const { return allNetPinIndices[netId]; }
  const utils::BoxT<int> &getBoundingBox(int netId) const { return allNetBoudingBoxes[netId]; }
  const cuda_shared_ptr<int[]> &getAllNetPinIndices() const { return devAllNetPinIndices; }
  const std::vector<int> &getAllNetPinIndicesOffset() const { return allNetPinIndicesOffset; }

  void updateCost(const utils::BoxT<int> &box);
  void reverse(int netId);
  void commit(int netId, int rootIdx, const cuda_shared_ptr<int[]> &routes);
  void getOverflowAndOpenIndices(std::vector<int> &netIndices) const;
  void applyToCpu(const std::vector<int> &netIndices);

private:
  std::shared_ptr<GRTreeNode> extractGRTree(const int *routes, int rootIdx) const;
  bool checkNetIsRouted(const GRNet &net) const;
  void selectAccessPoints(const GRNet &net, std::vector<int> &pinIndices) const;
  utils::BoxT<int> getPinIndicesBoundingBox(const std::vector<int> &pinIndices, int DIRECTION, int N) const;

private:
  std::vector<GRNet> &nets;
  GridGraph &gridGraph;
  const Parameters &parameters;

  int DIRECTION, N, X, Y, LAYER;
  int maxNumPins;
  
  std::vector<std::vector<int>> allNetPinIndices;
  std::vector<utils::BoxT<int>> allNetBoudingBoxes;
  std::vector<int> allNetPinIndicesOffset;
  cuda_shared_ptr<int[]> devAllNetPinIndices;

  realT unitLengthWireCost;
  realT unitViaCost;

  cuda_unique_ptr<realT[]> devHEdgeLengths;
  cuda_unique_ptr<realT[]> devVEdgeLengths;
  cuda_unique_ptr<realT[]> devUnitOverflowCosts;

  cuda_unique_ptr<int[]> devFlag;
  cuda_unique_ptr<realT[]> devCapacity;
  cuda_unique_ptr<realT[]> devDemand;
  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devNonStackViaCost;

  std::vector<int> allNetRootIndices;
  std::vector<int> allNetRoutes;
  cuda_unique_ptr<int[]> devAllNetRoutes;
  std::vector<int> allNetRoutesOffset;
  cuda_unique_ptr<int[]> devAllNetRoutesOffset;
};

#endif
#endif