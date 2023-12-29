#pragma once
#include "GridGraph.h"
#include "GRNet.h"
#include "../obj/ISPD24Parser.h"

class GlobalRouter {
public:
    GlobalRouter(const ISPD24Parser& parser, const Parameters& params);
    void route();
    void write(std::string guide_file = "");
    
private:
    const Parameters& parameters;
    GridGraph gridGraph;
    std::vector<GRNet> nets;
    
    int areaOfPinPatches;
    int areaOfWirePatches;

    // for evaluation
    CostT unit_length_wire_cost;
    CostT unit_via_cost;
    std::vector<CostT> unit_length_short_costs;
    
    void sortNetIndices(std::vector<int>& netIndices) const;
    void getGuides(const GRNet& net, std::vector<std::pair<int, utils::BoxT<int>>>& guides) ;
    void printStatistics() const;
};