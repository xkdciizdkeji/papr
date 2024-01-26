#pragma once
#include "GridGraph.h"
#include "GRNet.h"
#include "../obj/ISPD24Parser.h"
#include "../multithread/Scheduler.h" 
#include "../multithread/SingleNetRouter.h"

class GlobalRouter {
public:
    GlobalRouter(const ISPD24Parser& parser, const Parameters& params);
    void route();
    void write(std::string guide_file = "");

    void update_nonstack_via_counter(unsigned net_idx, const std::vector<std::vector<int>> &via_loc, std::vector<std::vector<std::vector<int>>> &flag, std::vector<std::vector<std::vector<int>>> &nonstack_via_counter) const;

private:
    const Parameters& parameters;
    GridGraph gridGraph;
    std::vector<GRNet> nets;
    
    int areaOfPinPatches;
    int areaOfWirePatches;

    int numofThreads;

    // for evaluation
    CostT unit_length_wire_cost;
    CostT unit_via_cost;
    std::vector<CostT> unit_length_short_costs;
    
    void sortNetIndices(std::vector<int>& netIndices) const; 
    void sortNetIndicesD(std::vector<int> &netIndices) const;
    void sortNetIndicesOFD(std::vector<int> &netIndices, std::vector<int> &netOverflow) const;
    void sortNetIndicesOFDALD(std::vector<int> &netIndices, std::vector<int> &netOverflow) const;
    void sortNetIndicesOFDALI(std::vector<int> &netIndices, std::vector<int> &netOverflow) const;
    void sortNetIndicesOLD(std::vector<int> &netIndices) const;
    void sortNetIndicesOLI(std::vector<int> &netIndices) const;
    void sortNetIndicesRandom(std::vector<int> &netIndices) const;
    void getGuides(const GRNet &net, std::vector<std::pair<std::pair<int, int>, utils::BoxT<int>>> &guides);
    // void getGuides(const GRNet &net, std::vector<std::pair<int, utils::BoxT<int>>> &guides);
    void printStatistics() const;

    void runJobsMT(int numJobs, int numofThreads, const std::function<void(int)>& handle);
    void runJobsMTnew(std::vector<std::vector<int>> batches, const std::function<void(int)>& handle);
    std::vector<std::vector<int>> getBatches(std::vector<SingleNetRouter>& routers, const std::vector<int>& netsToRoute);
};