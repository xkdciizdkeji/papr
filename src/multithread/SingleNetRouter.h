#pragma once
#include "../gr/GRNet.h"
#include "../obj/GeoTypes.h"

class SingleNetRouter {
public:
    GRNet& grNet;

    SingleNetRouter(GRNet& grDatabaseNet):grNet(grDatabaseNet){};

    std::vector<BoxOnLayer> guides;
};