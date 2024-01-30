#include "Scheduler.h"
#include <boost/geometry.hpp>

using namespace std;
constexpr unsigned int X = 0;
constexpr unsigned int Y = 1;

vector<vector<int>> &Scheduler::scheduleOrderEq(int numofThreads,  const vector<int>& netsToRoute) {

    if (numofThreads == 1) {
        // simple case
        for (int routerId : netsToRoute) batches.push_back({routerId});
    } else {
        // normal case
        vector<bool> assigned(routers.size(), false);
        int lastUnroute = 0;

        while (lastUnroute < netsToRoute.size()) {
            // create a new batch from a seed
            batches.emplace_back();
            initSet({});
            vector<int> &batch = batches.back();
            int lastMetric = INT_MAX;  // Note: need to be change if sort metric changes

            for (int i = lastUnroute; i < netsToRoute.size(); i++) {
                int routerId = i;

                if (assigned[routerId]) continue;

                int metric = routers[routerId].grNet.getBoundingBox().hp();
                if (metric > lastMetric) break;
                
                if (hasConflict(routerId)) {
                    lastMetric = metric;
                } else {
                    batch.push_back(netsToRoute[routerId]);
                    assigned[routerId] = true;
                    updateSet(routerId);
                }
            }

            // find the next seed
            while (lastUnroute < netsToRoute.size() && assigned[lastUnroute]) {
                ++lastUnroute;
            }
        }
    }
    return batches;
}

void Scheduler::initSet(vector<int> jobIdxes) {
    rtrees = RTrees(layer_num);
    for (int jobIdx : jobIdxes) {
        updateSet(jobIdx);
    }
}

void Scheduler::updateSet(int jobIdx) {
    for (const auto &guide : routers[jobIdx].guides) {
        boostBox box(boostPoint(guide[X].low, guide[Y].low), boostPoint(guide[X].high, guide[Y].high));
        rtrees[guide.layerIdx].insert({box, jobIdx});
    }
}

bool Scheduler::hasConflict(int jobIdx) {
    for (const auto &guide : routers[jobIdx].guides) {
        boostBox box(boostPoint(guide[X].low, guide[Y].low), boostPoint(guide[X].high, guide[Y].high));

        std::vector<std::pair<boostBox, int>> results;
        rtrees[guide.layerIdx].query(bgi::intersects(box), std::back_inserter(results));

        for (const auto &result : results) {
            if (result.second != jobIdx) {
                return true;
            }
        }
    }
    return false;
}