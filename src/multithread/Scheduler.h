#pragma once
#include "../global.h"
#include "SingleNetRouter.h"

class Scheduler {
public:
    Scheduler(const vector<SingleNetRouter>& routersToExec,int layer) : routers(routersToExec),layer_num(layer){}
    vector<vector<int>>& scheduleOrderEq(int numofThreads);
    int layer_num;

private:
    const vector<SingleNetRouter>& routers;
    vector<vector<int>> batches;

    // for conflict detect
    RTrees rtrees;
    void initSet(vector<int> jobIdxes);
    void updateSet(int jobIdx);
    bool hasConflict(int jobIdx);
};

