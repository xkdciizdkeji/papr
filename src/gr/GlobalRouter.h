#pragma once
#include "global.h"
#include "obj/Design.h"
#include "GridGraph.h"
#include "GRNet.h"

#include "multithread/Scheduler.h" 
#include "multithread/SingleNetRouter.h"

class GlobalRouter {
public:
    // GlobalRouter(const Design& design, const Parameters& params);
    GlobalRouter(const Parser& parser, const ParametersISPD24& params);
    void route();
    void netSortRoute(int cycles);
    void write(std::string guide_file = "");

private:
    // const Parameters& parameters;
    const ParametersISPD24& parameters;
    GridGraph gridGraph;
    vector<GRNet> nets;
    
    int areaOfPinPatches;
    int areaOfWirePatches;
    int numofThreads;
    
    void sortNetIndices(vector<int>& netIndices) const;

    void sortNetIndicesD(vector<int> &netIndices) const;

    void sortNetIndicesOFD(vector<int> &netIndices, vector<int> &netOverflow) const;
    void sortNetIndicesOFDALD(vector<int> &netIndices, vector<int> &netOverflow) const;
    void sortNetIndicesOLD(vector<int> &netIndices, vector<int> &netScores);
    void sortNetIndicesOLD(vector<int> &netIndices) const;
    void sortNetIndicesOLI(vector<int> &netIndices) const;
    void sortNetIndicesRandom(vector<int> &netIndices) const;
    void getGuides(const GRNet &net, vector<std::pair<int, utils::BoxT<int>>> &guides);
    void printStatistics() const;

    void runJobsMT(int numJobs, int numofThreads, const std::function<void(int)>& handle);
    vector<vector<int>> getBatches(vector<SingleNetRouter>& routers, const vector<int>& netsToRoute);
};