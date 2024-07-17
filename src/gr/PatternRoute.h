#pragma once
#include "global.h"
#include "GRNet.h"
#include "flute.h"


//Modified by IrisLin
#include "Torchroute.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include "GRTree.h"


extern "C" {
void readLUT();
Tree flute(int d, DTYPE x[], DTYPE y[], int acc);
}

class SteinerTreeNode: public utils::PointT<int> {
public:
    vector<std::shared_ptr<SteinerTreeNode>> children;
    utils::IntervalT<int> fixedLayers;//似乎steiner node没有这一项的信息
    
    SteinerTreeNode(utils::PointT<int> point): utils::PointT<int>(point) {}
    SteinerTreeNode(utils::PointT<int> point, utils::IntervalT<int> _fixedLayers): 
        utils::PointT<int>(point), fixedLayers(_fixedLayers) {}
        
    static void preorder(std::shared_ptr<SteinerTreeNode> node, std::function<void(std::shared_ptr<SteinerTreeNode>)> visit);
    static std::string getPythonString(std::shared_ptr<SteinerTreeNode> node);

    //Modified by IrisLin
    static std::vector<TorchEdge> getTorchEdges(std::shared_ptr<SteinerTreeNode> node);
};

class PatternRoutingNode: public utils::PointT<int> {
public:
    const int index;
    // int x
    // int y
    vector<std::shared_ptr<PatternRoutingNode>> children;//一个node可以有多个children
    vector<vector<std::shared_ptr<PatternRoutingNode>>> paths;//某个steiner node的child方向的某个下一个点，例如node1在steinertree里面有两个child，假如是node2和node3，paths[0][0]指向的node就是node1和node2之间的第一个点（指的是(node2.x,node1,y))，path[0][1]就是(node1.x,node2.y)。如果node1和node2在同一条线上，那paths[0][0]指的就是node2
    // childIndex -> pathIndex -> path
    utils::IntervalT<int> fixedLayers; 
    // layers that must be visited in order to connect all the pins 
    vector<CostT> costs; // layerIndex -> cost 
    vector<vector<std::pair<int, int>>> bestPaths; 
    // best path for each child; layerIndex -> childIndex -> (pathIndex, layerIndex)
    bool optional;
    
    PatternRoutingNode(utils::PointT<int> point, int _index, bool _optional = false): //这里的optional应该是那些可以移动的点，例如DAG的mid这种是可以移动的，然而pin这些是不能移动的。
        utils::PointT<int>(point), index(_index), optional(_optional) {}
    PatternRoutingNode(utils::PointT<int> point, utils::IntervalT<int> _fixedLayers, int _index = 0): 
        utils::PointT<int>(point), fixedLayers(_fixedLayers), index(_index), optional(false) {}
    static void preorder(std::shared_ptr<PatternRoutingNode> node, std::function<void(std::shared_ptr<PatternRoutingNode>)> visit);
    static std::string getPythonString(std::shared_ptr<PatternRoutingNode> routingDag);

    // //Modified by fengjx
    // static std::vector<std::vector<TorchEdge>> getTorchEdges(std::shared_ptr<PatternRoutingNode> node,robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> pin_locations);
    std::shared_ptr<PatternRoutingNode> buildPatternRoutingNode(std::shared_ptr<GRTreeNode> grNode, int& index);
    
};

class PatternRoute {
public:
    static void readFluteLUT() { readLUT(); };
    
    PatternRoute(GRNet& _net, const GridGraph& graph, const Parameters& param): 
        net(_net), gridGraph(graph), parameters(param), numDagNodes(0) {}
    void constructSteinerTree();
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> constructSteinerTree_return_selectedAccessPoints();
    void constructSteinerTree_Random();
    void constructSteinerTree_Random_multi(int treeNum);//可以同时建立多个树
    void constructSteinerTree_based_on_routingTree(std::shared_ptr<GRTreeNode>,robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints);
    void findAllTrees(std::shared_ptr<PatternRoutingNode> node, std::unordered_set<int>& specialNodes, std::vector<std::shared_ptr<PatternRoutingNode>>& path, std::vector<std::vector<std::shared_ptr<PatternRoutingNode>>>& results);
    void constructSteinerTree_based_on_routingTree63(std::shared_ptr<GRTreeNode>);
    void constructRoutingDAG();
    void constructRoutingDAG_based_on_routingTree(std::shared_ptr<GRTreeNode>);
    
    void constructDetours(GridGraphView<bool>& congestionView);
    void run();
    void setSteinerTree(std::shared_ptr<SteinerTreeNode> tree) { steinerTree = tree; }
    void setRoutingDAG(std::shared_ptr<PatternRoutingNode> dag) { routingDag = dag; }


    std::shared_ptr<SteinerTreeNode> getsT() { return steinerTree;}
    void clear_steinerTree() { steinerTree = nullptr;}
    std::shared_ptr<PatternRoutingNode> getrT() { return routingDag;}
    void clear_routingDag() { routingDag = nullptr;numDagNodes=0;}


    GRNet getNet(){return net;}


    //Modified by IrisLin&Feng
    void constructRoutingDAGfixed(torch::Tensor& ChosenPattern,int& TwoPinNetInedex);
    void constructRoutingDAGfixed55(vector<int>& ChosenPattern);
    void constructRoutingDAGfixed_again(torch::Tensor& ChosenPattern,int NetInedex);


    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> print_pin();
    
private:
    const Parameters& parameters;
    const GridGraph& gridGraph;
    GRNet& net;
    int numDagNodes;//应该是指DAG中node的数量
    std::shared_ptr<SteinerTreeNode> steinerTree;
    std::vector<std::shared_ptr<SteinerTreeNode>> steinerTree_multi;
    std::shared_ptr<PatternRoutingNode> routingDag;
    
    void constructPaths(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end, int childIndex = -1);
    void constructPaths(std::shared_ptr<GRTreeNode>& start, std::shared_ptr<GRTreeNode>& end, int childIndex = -1);
    //Modified by IrisLin&Feng
    void constructPathsfixed(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,torch::Tensor& ChosenPattern,int& TwoPinNetInedex, int childIndex = -1);
    void constructPathsfixed55(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,int& ChosenPattern,int& TwoPinNetInedex, int childIndex = -1);
    void constructPathsfixed_again(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,torch::Tensor& ChosenPattern,int NetIndex,int& TwoPinNetInedex, int childIndex = -1);
    void calculateRoutingCosts(std::shared_ptr<PatternRoutingNode>& node);
    std::shared_ptr<GRTreeNode> getRoutingTree(std::shared_ptr<PatternRoutingNode>& node, int parentLayerIndex = -1);
};


