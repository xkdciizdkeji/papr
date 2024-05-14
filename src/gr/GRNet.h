#pragma once
#include "global.h"
#include "obj/Design.h"
#include "GridGraph.h"
#include "Torchroute.h"

class GRNet {
public:
    GRNet(const Net& baseNet, const Design& design, const GridGraph& gridGraph);
    
    int getIndex() const { return index; }
    std::string getName() const { return name; }
    int getNumPins() const { return pinAccessPoints.size(); }
    const vector<vector<GRPoint>>& getPinAccessPoints() const { return pinAccessPoints; }
    const utils::BoxT<int>& getBoundingBox() const { return boundingBox; }
    const std::shared_ptr<GRTreeNode>& getRoutingTree() const { return routingTree; }
    // void getGuides(vector<std::pair<int, utils::BoxT<int>>>& guides) const;
    
    void setRoutingTree(std::shared_ptr<GRTreeNode> tree) { routingTree = tree; }
    void clearRoutingTree() { routingTree = nullptr; }

    std::vector<TorchEdge> getTorchEdges(std::shared_ptr<GRTreeNode> tree);

    static void preorder(std::shared_ptr<GRTreeNode> node, std::function<void(std::shared_ptr<GRTreeNode>)> visit);
    

private:
    int index;
    std::string name;
    vector<vector<GRPoint>> pinAccessPoints;
    utils::BoxT<int> boundingBox;
    std::shared_ptr<GRTreeNode> routingTree;
};