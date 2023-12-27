#include "GRNet.h"


GRNet::GRNet(const Net& baseNet, const Design& design, const GridGraph& gridGraph) {
    index = baseNet.getIndex();
    name = baseNet.getName();
    const auto& pinRefs = baseNet.getAllPinRefs();
    int numPins = pinRefs.size();
    pinAccessPoints.resize(numPins);
    for (int pinIndex = 0; pinIndex < pinRefs.size(); pinIndex++) {
        vector<BoxOnLayer> pinShapes;
        design.getPinShapes(pinRefs[pinIndex], pinShapes);
        robin_hood::unordered_set<uint64_t> included;
        for (const auto& pinShape : pinShapes) {
            utils::BoxT<int> cells = gridGraph.rangeSearchCells(pinShape);
            for (int x = cells.x.low; x <= cells.x.high; x++) {
                for (int y = cells.y.low; y <= cells.y.high; y++) {
                    GRPoint point(pinShape.layerIdx, x, y);
                    uint64_t hash = gridGraph.hashCell(point);
                    if (included.find(hash) == included.end()) {
                        pinAccessPoints[pinIndex].emplace_back(pinShape.layerIdx, x, y);
                        included.insert(hash);
                    }
                }
            }
        }
    }
    for (const auto& accessPoints : pinAccessPoints) {
        for (const auto& point : accessPoints) {
            boundingBox.Update(point);
        }
    }
}

GRNet::GRNet(PNet pnet)
{
    index = pnet.id;
    name = pnet.name;
    const auto& accessPointsList = pnet.accessPoints;
    int numPins = accessPointsList.size();
    // pinAccessPoints.resize(numPins);
    for (const auto& accessPoints: accessPointsList) {
        std::vector<GRPoint> pinAccessPoint;
        // log()<<"accessPoints.size():"<<accessPoints.size()<<std::endl;
        for (const auto& point : accessPoints) {
            // log() << "point: " << point[0] << " " << point[1] << " " << point[2] << std::endl;
            GRPoint grPoint(point[0], point[1], point[2]);
            pinAccessPoint.push_back(grPoint);
        }
        pinAccessPoints.push_back(pinAccessPoint);
    }
    // for(const auto& accessPoints: pinAccessPoints) {
    //     log()<<"accessPoints.size():"<<accessPoints.size()<<std::endl;
    //     for(const auto& point: accessPoints) {
    //         log() << "point: " << point.layerIdx << " " << point.x << " " << point.y << std::endl;
    //     }
    // }
    
    for (const auto& accessPoints : pinAccessPoints) {
        for (const auto& point : accessPoints) {
            boundingBox.Update(point);
        }
    }
    // log() <<"netid: "<<index << "boundingBox[lx]:"<< boundingBox.lx()<<"boundingBox[ly]:"<<boundingBox.ly()<<"boundingBox[hx]:"<<boundingBox.hx()<<"boundingBox[hy]:"<<boundingBox.hy()<<std::endl;
}

bool GRNet::overlap(GRNet net) const{
    utils::BoxT<int> otherBoundingBox= net.getBoundingBox();
    int lx1=boundingBox.lx();
    int ly1=boundingBox.ly();
    int ux1=boundingBox.hx();
    int uy1=boundingBox.hy();
    int lx2=otherBoundingBox.lx();
    int ly2=otherBoundingBox.ly();
    int ux2=otherBoundingBox.hx();
    int uy2=otherBoundingBox.hy();
    // Determine whether two rectangles intersect
    if (lx1 > ux2 || lx2 > ux1) return false;
    if (ly1 > uy2 || ly2 > uy1) return false;
    return true;
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