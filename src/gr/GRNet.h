#pragma once
// #include "../obj/Design.h"
// #include "GridGraph.h"

#include <vector>
#include <string>
#include "GRTree.h"
#include "../utils/utils.h"

class GRNet {
public:
    // GRNet(const Net& baseNet, const Design& design, const GridGraph& gridGraph);
    GRNet(int index, const std::string &name, const std::vector<std::vector<std::tuple<int, int, int>>> &pinAccessPoints);

    bool overlap(GRNet net) const;

    int getIndex() const { return index; }
    std::string getName() const { return name; }
    int getNumPins() const { return pinAccessPoints.size(); }
    const std::vector<std::vector<GRPoint>>& getPinAccessPoints() const { return pinAccessPoints; }
    const utils::BoxT<int>& getBoundingBox() const { return boundingBox; }
    const std::shared_ptr<GRTreeNode>& getRoutingTree() const { return routingTree; }
    // void getGuides(std::vector<std::pair<int, utils::BoxT<int>>>& guides) const;
    
    void setRoutingTree(std::shared_ptr<GRTreeNode> tree) { routingTree = tree; }
    void clearRoutingTree() { routingTree = nullptr; }
    
private:
    int index;
    std::string name;
    std::vector<std::vector<GRPoint>> pinAccessPoints; // 每个PIN有多个GRPoint选择
    utils::BoxT<int> boundingBox;
    std::shared_ptr<GRTreeNode> routingTree;
};