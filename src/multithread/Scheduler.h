#pragma once
#include "../utils/utils.h"
#include "SingleNetRouter.h"
#include <boost/geometry.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using boostPoint = bg::model::point<DBU, 2, bg::cs::cartesian>;
using boostBox = bg::model::box<boostPoint>;
using RTree = bgi::rtree<std::pair<boostBox, int>, bgi::linear<16>>; 
using RTrees = std::vector<bgi::rtree<std::pair<boostBox, int>, bgi::linear<16>>>;

class Scheduler {
public:
    Scheduler(const std::vector<SingleNetRouter>& routersToExec,int layer) : routers(routersToExec),layer_num(layer){}
    std::vector<std::vector<int>>& scheduleOrderEq(int numofThreads);
    int layer_num;

private:
    const std::vector<SingleNetRouter>& routers;
    std::vector<std::vector<int>> batches;

    // for conflict detect
    RTrees rtrees;
    void initSet(std::vector<int> jobIdxes);
    void updateSet(int jobIdx);
    bool hasConflict(int jobIdx);
};