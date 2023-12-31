#include "GeoTypes.h"

// BoxOnLayer

bool BoxOnLayer::isConnected(const BoxOnLayer& rhs) const {
    return abs(rhs.layerIdx - layerIdx) < 2 && HasIntersectWith(rhs);
}

std::ostream& operator<<(std::ostream& os, const BoxOnLayer& box) {
    os << "box(l=" << box.layerIdx << ", x=" << box[0] << ", y=" << box[1] << ")";
    return os;
}