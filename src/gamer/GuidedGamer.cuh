#ifdef ENABLE_CUDA
#ifndef GPU_ROUTER_GUIDED_GAMER_H
#define GPU_ROUTER_GUIDED_GAMER_H

#include "gamer_utils.cuh"

class GuidedGamer
{
public:
  GuidedGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins);

  void setWireCost(const cuda_shared_ptr<realT[]> &cost) { devWireCost = cost; }
  void setNonStackViaCost(const cuda_shared_ptr<realT[]> &cost) { devNonStackViaCost = cost; }
  void setUnitViaCost(realT cost) { unitViaCost = cost; }
  const cuda_shared_ptr<int[]> &getRoutes() const { return devRoutes; }
  bool getIsRouted() const;

  void setGuide2D(const std::vector<utils::BoxT<int>> &guide2D);
  void reserve(int nWires, int nRows, int nLongWires, int nWorkplace, int nViasegs);

  void route(const std::vector<int> &pinIndices, int numTurns);

private:
  int DIRECTION, N, X, Y, LAYER;
  int numPins, maxNumPins;

  int numWires, maxNumWires;
  int numRows, maxNumRows;
  int numLongWires, maxNumLongWires;
  int numWorkplace, maxNumWorkplace;
  int numViasegs, maxNumViasegs;
  int longWireEndRowsOffset;

  // pack
  cuda_unique_ptr<int3[]> devWirePackPlan;
  cuda_unique_ptr<int3[]> devViasegPackPlan;
  cuda_unique_ptr<int[]> devIdxPosMap;

  // route
  cuda_shared_ptr<int[]> devRoutes;
  cuda_unique_ptr<int[]> devIsRoutedPin;
  cuda_unique_ptr<int[]> devPinIndices;
  cuda_unique_ptr<int[]> devPinPositions;

  // wire row
  cuda_unique_ptr<int[]> devIdxAtRow;
  cuda_unique_ptr<int[]> devLocAtRow;
  cuda_unique_ptr<realT[]> devCostAtRow;
  cuda_unique_ptr<realT[]> devDistAtRow;
  cuda_unique_ptr<int[]> devAllPrevAtRow;
  cuda_unique_ptr<int[]> devMarkAtRow;

  // via segment
  cuda_unique_ptr<int[]> devPosAtViaseg;
  cuda_unique_ptr<realT[]> devCostAtViaseg;

  // long wire row & their workplace
  cuda_unique_ptr<int[]> devLongWireOffsets;
  cuda_unique_ptr<int[]> devLongWireLengths;
  cuda_unique_ptr<char[]> devWorkplace;

  // original cost
  cuda_shared_ptr<realT[]> devWireCost;
  cuda_shared_ptr<realT[]> devNonStackViaCost;
  realT unitViaCost;
};

#endif
#endif