#ifdef ENABLE_CUDA
#ifndef GPU_ROUTER_GUIDED_GAMER_H
#define GPU_ROUTER_GUIDED_GAMER_H

#include "gamer_utils.cuh"

class GuidedGamer
{
public:
  GuidedGamer(int DIRECTION, int N, int X, int Y, int LAYER, int maxNumPins);

  void setWireCostMap(const cuda_shared_ptr<const realT[]> &wireCost) { devWireCost = wireCost; }
  void setViaCostMap(const cuda_shared_ptr<const realT[]> &viaCost) { devViaCost = viaCost; }
  cuda_shared_ptr<const int[]> getRoutes() const { return devRoutes; }
  bool getIsRouted() const;

  void setGuide(const int *routes, int scaleX, int scaleY, int coarseN);
  void reserve(int nWires, int nRows, int nLongWires, int nWorkplace, int nViasegs);

  void route(const std::vector<int> &pinIndices, int sweepTurns);

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
  cuda_unique_ptr<int[]> devPrevAtRow;

  // via segment
  cuda_unique_ptr<int[]> devPosAtViaseg;
  cuda_unique_ptr<realT[]> devCostAtViaseg;

  // long wire row & their workplace
  cuda_unique_ptr<int[]> devLongWireOffsets;
  cuda_unique_ptr<int[]> devLongWireLengths;
  cuda_unique_ptr<char[]> devWorkplace; // we should allocate `N/ROW_SIZE*ROW_SIZE` of memory for each long row

  // original cost
  cuda_shared_ptr<const realT[]> devWireCost;
  cuda_shared_ptr<const realT[]> devViaCost;
};

#endif
#endif