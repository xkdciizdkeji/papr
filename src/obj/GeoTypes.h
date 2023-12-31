#pragma once
#include "../utils/utils.h"

class BoxOnLayer: public utils::BoxT<DBU> {
public:
    int layerIdx;

    //  constructors
    template <typename... Args>
    BoxOnLayer(int layerIndex = -1, Args... params) : layerIdx(layerIndex), utils::BoxT<DBU>(params...) {}

    // inherit setters from utils::BoxT in batch
    template <typename... Args>
    void Set(int layerIndex = -1, Args... params) {
        layerIdx = layerIndex;
        utils::BoxT<DBU>::Set(params...);
    }

    bool isConnected(const BoxOnLayer& rhs) const;

    friend std::ostream& operator<<(std::ostream& os, const BoxOnLayer& box);
};