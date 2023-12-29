#include "GRNet.h"

// using std::vector;
// GRNet::GRNet(const Net& baseNet, const Design& design, const GridGraph& gridGraph) {
//     index = baseNet.getIndex();
//     name = baseNet.getName();
//     const auto& pinRefs = baseNet.getAllPinRefs();
//     int numPins = pinRefs.size();
//     pinAccessPoints.resize(numPins);
//     for (int pinIndex = 0; pinIndex < pinRefs.size(); pinIndex++) {
//         vector<BoxOnLayer> pinShapes;
//         design.getPinShapes(pinRefs[pinIndex], pinShapes);
//         robin_hood::unordered_set<uint64_t> included;
//         for (const auto& pinShape : pinShapes) {
//             utils::BoxT<int> cells = gridGraph.rangeSearchCells(pinShape);
//             for (int x = cells.x.low; x <= cells.x.high; x++) {
//                 for (int y = cells.y.low; y <= cells.y.high; y++) {
//                     GRPoint point(pinShape.layerIdx, x, y);
//                     uint64_t hash = gridGraph.hashCell(point);
//                     if (included.find(hash) == included.end()) {
//                         pinAccessPoints[pinIndex].emplace_back(pinShape.layerIdx, x, y);
//                         included.insert(hash);
//                     }
//                 }
//             }
//         }
//     }
//     for (const auto& accessPoints : pinAccessPoints) {
//         for (const auto& point : accessPoints) {
//             boundingBox.Update(point);
//         }
//     }
// }

GRNet::GRNet(int _index, const std::string &_name, const std::vector<std::vector<std::tuple<int, int, int>>> &_pinAccessPoints)
    : index(_index), name(_name)
{
    pinAccessPoints.resize(_pinAccessPoints.size());
    for(int i = 0; i < pinAccessPoints.size(); i++)
    {
        pinAccessPoints[i].reserve(_pinAccessPoints[i].size());
        for(auto [x, y, z] : _pinAccessPoints[i])
            pinAccessPoints[i].emplace_back(z, x, y);
    }
    for (const auto &accessPoints : pinAccessPoints)
        for (const auto &point : accessPoints)
            boundingBox.Update(point);
}

// void GRNet::getGuides(vector<std::pair<int, utils::BoxT<int>>>& guides) const {
//     if (!routingTree) return;
//     GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
//         for (const auto& child : node->children) {
//             if (node->layerIdx == child->layerIdx) {
//                 guides.emplace_back(
//                     node->layerIdx, utils::BoxT<int>(
//                         min(node->x, child->x), min(node->y, child->y),
//                         max(node->x, child->x), max(node->y, child->y)
//                     )
//                 );
//             } else {
//                 int maxLayerIndex = max(node->layerIdx, child->layerIdx);
//                 for (int layerIdx = min(node->layerIdx, child->layerIdx); layerIdx <= maxLayerIndex; layerIdx++) {
//                     guides.emplace_back(layerIdx, utils::BoxT<int>(node->x, node->y));
//                 }
//             }
//         }
//     });
// }