#include "PatternRoute.h"
#include <random>  // 包含 <random> 头文件
#include "GRTree.h"
using std::vector;

void SteinerTreeNode::preorder(
    std::shared_ptr<SteinerTreeNode> node, 
    std::function<void(std::shared_ptr<SteinerTreeNode>)> visit
) {
    visit(node);
    for (auto& child : node->children) preorder(child, visit);
}

void PatternRoutingNode::preorder(
    std::shared_ptr<PatternRoutingNode> node, 
    std::function<void(std::shared_ptr<PatternRoutingNode>)> visit
) {
    visit(node);
    for (auto& child : node->children) preorder(child, visit);
}




// Modified by IrisLin
void TorchPin::set(int xloc, int yloc, int l)
{
    x = xloc;
    y = yloc;
    layer = l;
}

void TorchEdge::set(TorchPin pintemp1, TorchPin pintemp2)
{
    pin1 = pintemp1;
    pin2 = pintemp2;
}

// Modified by IrisLin
std::vector<TorchEdge> SteinerTreeNode::getTorchEdges(std::shared_ptr<SteinerTreeNode> node)
{
    std::vector<TorchEdge> edges;
    preorder(node, [&](std::shared_ptr<SteinerTreeNode> node)
             {
        for (auto child : node->children) {
            TorchPin pin1,pin2;
            pin1.set(node->x, node->y, 0);
            pin2.set(child->x, child->y, 0);
            //if (node->x==child->x && node->y==child->y) std::cout<<"yes ";
            TorchEdge edge;
            edge.set(pin1, pin2);
            edges.emplace_back(edge);
        } });
    return edges;
}

// std::vector<std::vector<TorchEdge>> PatternRoutingNode::getTorchEdges(std::shared_ptr<PatternRoutingNode> node,robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> pin_locations)
// {
//     std::vector<std::vector<TorchEdge>> trees;
//     auto treeConnectsSpecialPoints = [&pin_locations](std::shared_ptr<PatternRoutingNode> treeRoot) {
//         for (const auto& path : treeRoot->paths) {
//             bool containsSpecialPoints = false;
//             for (const auto& node : path) {
//                 for (const auto& specialPoint : pin_locations) {
//                     if (node->children == specialPoint) {
//                         containsSpecialPoints = true;
//                         break;
//                     }
//                 }
//                 if (containsSpecialPoints) break;
//             }
//             if (containsSpecialPoints) return true;
//         }
//         return false;
//     };
//     preorder(node, [&trees, &pin_locations, &treeConnectsSpecialPoints](std::shared_ptr<PatternRoutingNode> currentNode) {
//         if (treeConnectsSpecialPoints(currentNode)) {
//             // 如果当前树结构连接了特殊点，将其转换为 TorchEdges 并添加到结果中
//             std::vector<TorchEdge> torchEdges;
//             for (const auto& path : currentNode->paths) {
//                 for (size_t i = 0; i < path.size() - 1; ++i) {
//                     TorchEdge edge;
//                     edge.set(path[i]->pin, path[i + 1]->pin);
//                     torchEdges.push_back(edge);
//                 }
//             }
//             trees.push_back(torchEdges);
//         }
//     });

// }

robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>>  PatternRoute::print_pin(){
    // blockage_net_count++;
    // 1. Select access points
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints;
    gridGraph.selectAccessPoints(net, selectedAccessPoints);

    // 2. Construct Steiner tree
    const int degree = selectedAccessPoints.size();
    if (degree != 1) {
        //std::cout<<"********************* "<<blockage_net_count<<" **********************"<<std::endl;
        for (auto& accessPoint : selectedAccessPoints){
            //std::cout<<"["<<accessPoint.second.first.x<<","<<accessPoint.second.first.y<<"]"<<std::endl;
        }
    }
    //loghline();
    return selectedAccessPoints;
}


std::string SteinerTreeNode::getPythonString(std::shared_ptr<SteinerTreeNode> node) {
    vector<std::pair<utils::PointT<int>, utils::PointT<int>>> edges;
    preorder(node, [&] (std::shared_ptr<SteinerTreeNode> node) {
        for (auto child : node->children) {
            edges.emplace_back(*node, *child);
        }
    });
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < edges.size(); i++) {
        auto& edge = edges[i];
        ss << "[" << edge.first << ", " << edge.second << "]";
        ss << (i < edges.size() - 1 ? ", " : "]");
    }
    return ss.str();
}

std::string PatternRoutingNode::getPythonString(std::shared_ptr<PatternRoutingNode> routingDag) {
    vector<std::pair<utils::PointT<int>, utils::PointT<int>>> edges;
    std::function<void(std::shared_ptr<PatternRoutingNode>)> getEdges = 
        [&] (std::shared_ptr<PatternRoutingNode> node) {
            for (auto& childPaths : node->paths) {
                for (auto& path : childPaths) {
                    edges.emplace_back(*node, *path);
                    getEdges(path);
                }
            }
        };
    getEdges(routingDag);
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < edges.size(); i++) {
        auto& edge = edges[i];
        ss << "[" << edge.first << ", " << edge.second << "]";
        ss << (i < edges.size() - 1 ? ", " : "]");
    }
    return ss.str();
}

void PatternRoute::constructSteinerTree() {
    // 1. Select access points
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints;
    gridGraph.selectAccessPoints(net, selectedAccessPoints);
    
    // 2. Construct Steiner tree
    const int degree = selectedAccessPoints.size();
    if (degree == 1) {
        for (auto& accessPoint : selectedAccessPoints) {
            steinerTree = std::make_shared<SteinerTreeNode>(accessPoint.second.first, accessPoint.second.second);
        }
    } else {
        int xs[degree * 4];
        int ys[degree * 4];
        int i = 0;
        for (auto& accessPoint : selectedAccessPoints) {
            xs[i] = accessPoint.second.first.x;
            ys[i] = accessPoint.second.first.y;
            i++;
        }
        Tree flutetree = flute(degree, xs, ys, ACCURACY);
        const int numBranches = degree + degree - 2;
        vector<utils::PointT<int>> steinerPoints;
        steinerPoints.reserve(numBranches);
        vector<vector<int>> adjacentList(numBranches);
        for (int branchIndex = 0; branchIndex < numBranches; branchIndex++) {
            const Branch& branch = flutetree.branch[branchIndex];
            steinerPoints.emplace_back(branch.x, branch.y);
            if (branchIndex == branch.n) continue;
            adjacentList[branchIndex].push_back(branch.n);
            adjacentList[branch.n].push_back(branchIndex);
        }
        std::function<void(std::shared_ptr<SteinerTreeNode>&, int, int)> constructTree = [&] (
            std::shared_ptr<SteinerTreeNode>& parent, int prevIndex, int curIndex
        ) {
            std::shared_ptr<SteinerTreeNode> current = std::make_shared<SteinerTreeNode>(steinerPoints[curIndex]);
            if (parent != nullptr && parent->x == current->x && parent->y == current->y) {
                for (int nextIndex : adjacentList[curIndex]) {
                    if (nextIndex == prevIndex) continue;
                    constructTree(parent, curIndex, nextIndex);
                }
                return;
            }
            // Build subtree
            for (int nextIndex : adjacentList[curIndex]) {
                if (nextIndex == prevIndex) continue;
                constructTree(current, curIndex, nextIndex);
            }
            // Set fixed layer interval
            uint64_t hash = gridGraph.hashCell(current->x, current->y);
            if (selectedAccessPoints.find(hash) != selectedAccessPoints.end()) {
                current->fixedLayers = selectedAccessPoints[hash].second;
            }
            // Connect current to parent
            if (parent == nullptr) {
                parent = current;
            } else {
                parent->children.emplace_back(current);
            }
        };
        // Pick a root having degree 1
        int root = 0;
        std::function<bool(int)> hasDegree1 = [&] (int index) {
            if (adjacentList[index].size() == 1) {
                int nextIndex = adjacentList[index][0];
                if (steinerPoints[index] == steinerPoints[nextIndex]) {
                    return hasDegree1(nextIndex);
                } else {
                    return true;
                }
            } else {
                return false;
            }
        };
        for (int i = 0; i < steinerPoints.size(); i++) {
            if (hasDegree1(i)) {
                root = i;
                break;
            }
        }
        constructTree(steinerTree, -1, root);
    }
}
void PatternRoute::constructSteinerTree_Random() {
    // 1. Select access points
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints;
    gridGraph.selectAccessPoints(net, selectedAccessPoints);
    
    // 2. Construct Steiner tree
    const int degree = selectedAccessPoints.size();
    if (degree == 1) {
        for (auto& accessPoint : selectedAccessPoints) {
            steinerTree = std::make_shared<SteinerTreeNode>(accessPoint.second.first, accessPoint.second.second);
        }
    } else {
        int xs[degree * 4];
        int ys[degree * 4];
        int i = 0;
        for (auto& accessPoint : selectedAccessPoints) {
            xs[i] = accessPoint.second.first.x;
            ys[i] = accessPoint.second.first.y;
            i++;
        }
        Tree flutetree = flute(degree, xs, ys, ACCURACY);
        const int numBranches = degree + degree - 2;
        vector<utils::PointT<int>> steinerPoints;
        steinerPoints.reserve(numBranches);
        vector<vector<int>> adjacentList(numBranches);
        for (int branchIndex = 0; branchIndex < numBranches; branchIndex++) {
            const Branch& branch = flutetree.branch[branchIndex];
            steinerPoints.emplace_back(branch.x, branch.y);
            if (branchIndex == branch.n) continue;
            adjacentList[branchIndex].push_back(branch.n);
            adjacentList[branch.n].push_back(branchIndex);
        }
        std::function<void(std::shared_ptr<SteinerTreeNode>&, int, int)> constructTree = [&] (
            std::shared_ptr<SteinerTreeNode>& parent, int prevIndex, int curIndex
        ) {
            std::shared_ptr<SteinerTreeNode> current = std::make_shared<SteinerTreeNode>(steinerPoints[curIndex]);
            if (parent != nullptr && parent->x == current->x && parent->y == current->y) {
                for (int nextIndex : adjacentList[curIndex]) {
                    if (nextIndex == prevIndex) continue;
                    constructTree(parent, curIndex, nextIndex);
                }
                return;
            }
            // Build subtree
            for (int nextIndex : adjacentList[curIndex]) {
                if (nextIndex == prevIndex) continue;
                constructTree(current, curIndex, nextIndex);
            }
            // Set fixed layer interval
            uint64_t hash = gridGraph.hashCell(current->x, current->y);
            if (selectedAccessPoints.find(hash) != selectedAccessPoints.end()) {
                current->fixedLayers = selectedAccessPoints[hash].second;
            }
            // Connect current to parent
            if (parent == nullptr) {
                parent = current;
            } else {
                parent->children.emplace_back(current);
            }
        };
        // Pick a root having degree 1
        // 这回我随机选一个root的degree为1的点
        int root = 0;
        std::function<bool(int)> hasDegree1 = [&] (int index) {
            if (adjacentList[index].size() == 1) {
                int nextIndex = adjacentList[index][0];
                if (steinerPoints[index] == steinerPoints[nextIndex]) {
                    return hasDegree1(nextIndex);
                } else {
                    return true;
                }
            } else {
                return false;
            }
        };
        vector<int> pinindextemp(steinerPoints.size());
        for (int i = 0; i < steinerPoints.size(); i++) pinindextemp[i]=i;
        // 获取随机数生成器
        std::random_device rd;
        std::mt19937 g(rd());

        // 打乱向量顺序
        std::shuffle(pinindextemp.begin(), pinindextemp.end(), g);
        for (int pinindex:pinindextemp) {
            if (hasDegree1(pinindex)) {
                root = pinindex;
                break;
            }
        }
        constructTree(steinerTree, -1, root);
    }
}
void PatternRoute::constructSteinerTree_Random_multi(int treeNum) {
    steinerTree_multi.reserve(treeNum);
    // 1. Select access points
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints;
    gridGraph.selectAccessPoints(net, selectedAccessPoints);
    
    // 2. Construct Steiner tree
    const int degree = selectedAccessPoints.size();
    if (degree == 1) {
        for (auto& accessPoint : selectedAccessPoints) {
            for(int treecount=0;treecount<treeNum;treecount++) steinerTree_multi[treecount] = std::make_shared<SteinerTreeNode>(accessPoint.second.first, accessPoint.second.second);
        }
    } else {
        int xs[degree * 4];
        int ys[degree * 4];
        int i = 0;
        for (auto& accessPoint : selectedAccessPoints) {
            xs[i] = accessPoint.second.first.x;
            ys[i] = accessPoint.second.first.y;
            i++;
        }
        Tree flutetree = flute(degree, xs, ys, ACCURACY);
        const int numBranches = degree + degree - 2;
        vector<utils::PointT<int>> steinerPoints;
        steinerPoints.reserve(numBranches);
        vector<vector<int>> adjacentList(numBranches);
        for (int branchIndex = 0; branchIndex < numBranches; branchIndex++) {
            const Branch& branch = flutetree.branch[branchIndex];
            steinerPoints.emplace_back(branch.x, branch.y);
            if (branchIndex == branch.n) continue;
            adjacentList[branchIndex].push_back(branch.n);
            adjacentList[branch.n].push_back(branchIndex);
        }
        std::function<void(std::shared_ptr<SteinerTreeNode>&, int, int)> constructTree = [&] (
            std::shared_ptr<SteinerTreeNode>& parent, int prevIndex, int curIndex
        ) {
            std::shared_ptr<SteinerTreeNode> current = std::make_shared<SteinerTreeNode>(steinerPoints[curIndex]);
            if (parent != nullptr && parent->x == current->x && parent->y == current->y) {
                for (int nextIndex : adjacentList[curIndex]) {
                    if (nextIndex == prevIndex) continue;
                    constructTree(parent, curIndex, nextIndex);
                }
                return;
            }
            // Build subtree
            for (int nextIndex : adjacentList[curIndex]) {
                if (nextIndex == prevIndex) continue;
                constructTree(current, curIndex, nextIndex);
            }
            // Set fixed layer interval
            uint64_t hash = gridGraph.hashCell(current->x, current->y);
            if (selectedAccessPoints.find(hash) != selectedAccessPoints.end()) {
                current->fixedLayers = selectedAccessPoints[hash].second;
            }
            // Connect current to parent
            if (parent == nullptr) {
                parent = current;
            } else {
                parent->children.emplace_back(current);
            }
        };
        // Pick a root having degree 1
        // 这回我随机选一个root的degree为1的点
        int root = 0;
        vector<int> root_multi(steinerPoints.size());
        std::function<bool(int)> hasDegree1 = [&] (int index) {
            if (adjacentList[index].size() == 1) {
                int nextIndex = adjacentList[index][0];
                if (steinerPoints[index] == steinerPoints[nextIndex]) {
                    return hasDegree1(nextIndex);
                } else {
                    return true;
                }
            } else {
                return false;
            }
        };
        for (int i = 0; i < steinerPoints.size(); i++) {
            if (hasDegree1(i)) {
                root_multi[i]=i;//.push_back(i);
                break;
            }
        }
        //constructTree(steinerTree_multi[0], -1, root_multi[0]);

        vector<int> pinindextemp(steinerPoints.size());
        for (int i = 0; i < steinerPoints.size(); i++) pinindextemp[i]=i;
        // 获取随机数生成器
        std::random_device rd;
        std::mt19937 g(rd());

        // 打乱向量顺序
        std::shuffle(pinindextemp.begin(), pinindextemp.end(), g);

        for (int pinindex:pinindextemp) {
            if (hasDegree1(pinindex)) {
                root_multi.push_back(pinindex);
                if (root_multi.size()==treeNum) break;
            }
        }
        int root_multi_size_temp=root_multi.size();
        for(int i=0;root_multi.size()<treeNum;++i){
            root_multi.push_back(root_multi[i % root_multi_size_temp]);
        }
        for (int treecount=0;treecount<treeNum;++treecount) constructTree(steinerTree_multi[treecount], -1, root_multi[treecount]);
    }
}
void PatternRoute::constructSteinerTree_based_on_routingTree(std::shared_ptr<GRTreeNode> routingTree) {
    //getsT() = nullptr;
    std::function<void(std::shared_ptr<SteinerTreeNode>&, std::shared_ptr<GRTreeNode>&)> construct_SteinerTree = [&] (
        std::shared_ptr<SteinerTreeNode>& dstNode, std::shared_ptr<GRTreeNode>& gr
    ) {
        std::shared_ptr<SteinerTreeNode> current = std::make_shared<SteinerTreeNode>(
            //*gr, gr->layerIdx,numDagNodes++
            *gr, gr->layerIdx
        );
        for (auto grChild : gr->children) {
            construct_SteinerTree(current, grChild);
        }
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            dstNode->children.emplace_back(current);
            //constructPaths(dstNode, current);
        }
    };
    construct_SteinerTree(steinerTree, routingTree);

    steinerTree->preorder(steinerTree, [&](std::shared_ptr<SteinerTreeNode> visit) {
        //log() << *visit << (visit->children.size() > 0 ? " -> " : "");
        // for (auto& child : visit->children) {
        //     if(child->x==visit->x&&child->y==visit->y) {
        //     // if(*child==*visit) {
        //         for (auto& grandchild:child->children) visit->children.emplace_back(grandchild);
        //         child=nullptr;
        //         visit->children.erase(std::remove(visit->children.begin(), visit->children.end(), child), visit->children.end());
        //     }
        // }
        std::vector<int> remove_childindex;
        for (int childindex=0;childindex<visit->children.size();childindex++) {
            if(visit->children[childindex]->x==visit->x&&visit->children[childindex]->y==visit->y) {
            // if(*child==*visit) {
                for (auto& grandchild:visit->children[childindex]->children) visit->children.emplace_back(grandchild);
                remove_childindex.emplace_back(childindex);
                // visit->children.erase(visit->children.begin()+childindex);
                //visit->children.erase(std::remove(visit->children.begin(), visit->children.end(), child), visit->children.end());
            }
        }
        for (int remove_childindex_i=remove_childindex.size()-1;remove_childindex_i>=0;--remove_childindex_i) {
            visit->children.erase(visit->children.begin()+remove_childindex[remove_childindex_i]);
        }
        //std::cout << *child << (child == visit->children.back() ? "" : ", ");
        //std::cout << std::endl;
    });
}



void PatternRoute::constructSteinerTree_based_on_routingTree63(std::shared_ptr<GRTreeNode> node) {
    node->preorder(node, [](std::shared_ptr<GRTreeNode> visit) {
        log() << *visit << (visit->children.size() > 0 ? " -> " : "");
        std::shared_ptr<SteinerTreeNode> vi=std::make_shared<SteinerTreeNode>(*visit, visit->layerIdx);
        for (auto& child : visit->children) std::cout << *child << (child == visit->children.back() ? "" : ", ");
        
        
        std::cout << std::endl;
    });
}
void PatternRoute::constructRoutingDAG() {
    std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<SteinerTreeNode>&)> constructDag = [&] (
        std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<SteinerTreeNode>& steiner
    ) {
        std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
            *steiner, steiner->fixedLayers, numDagNodes++
        );
        for (auto steinerChild : steiner->children) {
            constructDag(current, steinerChild);
        }
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            dstNode->children.emplace_back(current);
            constructPaths(dstNode, current);
        }
    };
    constructDag(routingDag, steinerTree);
}

// void PatternRoute::constructRoutingDAG_based_on_routingTree(std::shared_ptr<GRTreeNode> routingTree) {
//     numDagNodes=0;
//     std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<GRTreeNode>&, std::shared_ptr<SteinerTreeNode>&)> constructDag = [&] (
//         std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<GRTreeNode>& gr,std::shared_ptr<SteinerTreeNode> steiner
//     ) {
//         std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
//             *gr, gr->layerIdx,numDagNodes++
//         );
//         std::shared_ptr<SteinerTreeNode> current_steiner = std::make_shared<SteinerTreeNode>(
//             *gr, gr->layerIdx
//         );
        
        
//         for (auto grChild : gr->children) {
//             constructDag(current, grChild, current_steiner);
//         }
//         if (dstNode == nullptr) {
//             dstNode = current;
//         } else {
//             dstNode->children.emplace_back(current);
//             constructPaths(dstNode, current);
//         }
//     };
//     constructDag(routingDag, routingTree,steinerTree);
//     //net.clearRoutingTree();
// }


void PatternRoute::constructRoutingDAG_based_on_routingTree(std::shared_ptr<GRTreeNode> routingTree) {
    numDagNodes=0;
    std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<GRTreeNode>&)> constructDag = [&] (
        std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<GRTreeNode>& gr
    ) {
        std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
            *gr, gr->layerIdx,numDagNodes++
        );
        for (auto grChild : gr->children) {
            constructDag(current, grChild);
        }
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            dstNode->children.emplace_back(current);
            constructPaths(dstNode, current);
        }
    };
    constructDag(routingDag, routingTree);
    //net.clearRoutingTree();
}
// void PatternRoute::constructRoutingDAG_based_on_routingTree(std::shared_ptr<GRTreeNode> routingTree) {
//     //numDagNodes=0;
//     //routingDag = nullptr;
//     std::cout<<"1"<<std::endl;

//     getrT()=nullptr;
//     std::cout<<"2"<<std::endl;
//     //将routingDag设定成与routingTree相同
//     GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
//         for (const auto& child : node->children) {
//             std::shared_ptr<PatternRoutingNode> patternroutingnode_node = std::make_shared<PatternRoutingNode>(*node,numDagNodes++,true);
//             std::shared_ptr<PatternRoutingNode> patterntoutingnode_child = std::make_shared<PatternRoutingNode>(*child,numDagNodes++,true);
//             constructPaths(patternroutingnode_node, patterntoutingnode_child);
//         }
//     });
   
// }

std::shared_ptr<PatternRoutingNode> PatternRoutingNode::buildPatternRoutingNode(std::shared_ptr<GRTreeNode> grNode, int& index) {
    auto patternNode = std::make_shared<PatternRoutingNode>(PointT<int>(grNode->x, grNode->y), index++);
    for (const auto& child : grNode->children) {
        patternNode->children.push_back(buildPatternRoutingNode(child, index));
    }
    return patternNode;
}

void PatternRoute::constructRoutingDAGfixed(torch::Tensor& ChoosedPatternIndexArray,int& TwoPinNetInedex) {
    // 定义一个函数，用于构建路由DAG
    //std::ofstream outf;    //用ofstream类定义输入对象
    //outf.open("a.txt", std::ios::app);
    //outf.open("a.txt");
    std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<SteinerTreeNode>&)> constructDag = [&] (
        std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<SteinerTreeNode>& steiner//定义一个函数对象，该对象接受两个参数：一个指向PatternRoutingNode的shared_ptr和一个指向SteinerTreeNode的shared_ptr
    ) {
        // 创建一个新的节点
        std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
            *steiner, steiner->fixedLayers, numDagNodes++
        );
        // 遍历steiner的子节点
        for (auto steinerChild : steiner->children) {
            // 递归调用函数，构建路由DAG
            constructDag(current, steinerChild);
        }
        // 如果dstNode为空，则将当前节点赋值给dstNode
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            // 否则将当前节点添加到dstNode的子节点中
            dstNode->children.emplace_back(current);
            // 构建路径
            constructPathsfixed(dstNode, current,ChoosedPatternIndexArray,TwoPinNetInedex);
            //std::cout<<dstNode->x<<" "<<dstNode->y<<std::endl;
            //std::cout<<*dstNode.getPythonString(routingDag)<<std::endl;
        }
    };
    
    // 调用函数，构建路由DAG
    constructDag(routingDag, steinerTree);
    //outf<<routingDag->getPythonString(routingDag)<<std::endl;
    // outf<<steinerTree->getPythonString(steinerTree)<<std::endl;
    //outf.close();
    //std::cout<<routingDag->getPythonString(routingDag)<<std::endl;
    //std::cout<<steinerTree->x<<" "<<steinerTree->y<<std::endl;
    
}
void PatternRoute::constructRoutingDAGfixed55(vector<int>& ChoosedPatternIndexArray) {
    // 定义一个函数，用于构建路由DAG
    //std::ofstream outf;    //用ofstream类定义输入对象
    //outf.open("a.txt", std::ios::app);
    //outf.open("a.txt");
    int TwoPinNetInedex=0;
    
    std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<SteinerTreeNode>&)> constructDag = [&] (
        std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<SteinerTreeNode>& steiner//定义一个函数对象，该对象接受两个参数：一个指向PatternRoutingNode的shared_ptr和一个指向SteinerTreeNode的shared_ptr
    ) {
        // 创建一个新的节点
        std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
            *steiner, steiner->fixedLayers, numDagNodes++
        );
        // 遍历steiner的子节点
        for (auto steinerChild : steiner->children) {
            // 递归调用函数，构建路由DAG
            constructDag(current, steinerChild);
        }
        // 如果dstNode为空，则将当前节点赋值给dstNode
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            // 否则将当前节点添加到dstNode的子节点中
            dstNode->children.emplace_back(current);
            // 构建路径
            //std::cout<<"TWOPINNETINDEX"<<TwoPinNetInedex<<std::endl;
            //std::cout<<ChoosedPatternIndexArray[TwoPinNetInedex]<<" ";
            constructPathsfixed55(dstNode, current,ChoosedPatternIndexArray[TwoPinNetInedex],TwoPinNetInedex);
            
            //TwoPinNetInedex++;
            //std::cout<<dstNode->x<<" "<<dstNode->y<<std::endl;
            //std::cout<<*dstNode.getPythonString(routingDag)<<std::endl;
        }
    };
    
    // 调用函数，构建路由DAG
    //log()<<ChoosedPatternIndexArray.size()<<std::endl;
    constructDag(routingDag, steinerTree);
    //if (TwoPinNetInedex==ChoosedPatternIndexArray.size()) std::cout<<"TwoPinNetInedex"<<TwoPinNetInedex<<"ChoosedPatternIndexArray.size()"<<ChoosedPatternIndexArray.size()<<"bad "<<std::endl;
    //outf<<routingDag->getPythonString(routingDag)<<std::endl;
    // outf<<steinerTree->getPythonString(steinerTree)<<std::endl;
    //outf.close();
    //std::cout<<routingDag->getPythonString(routingDag)<<std::endl;
    //std::cout<<steinerTree->x<<" "<<steinerTree->y<<std::endl;
    
}
void PatternRoute::constructRoutingDAGfixed_again(torch::Tensor& ChoosedPatternIndexArray,int NetInedex) {
    // 定义一个函数，用于构建路由DAG
    //std::ofstream outf;    //用ofstream类定义输入对象
    //outf.open("a.txt", std::ios::app);
    //outf.open("a.txt");
    int this_tree_two_pin_net_index = 0;
    std::function<void(std::shared_ptr<PatternRoutingNode>&, std::shared_ptr<SteinerTreeNode>&)> constructDag = [&] (
        std::shared_ptr<PatternRoutingNode>& dstNode, std::shared_ptr<SteinerTreeNode>& steiner//定义一个函数对象，该对象接受两个参数：一个指向PatternRoutingNode的shared_ptr和一个指向SteinerTreeNode的shared_ptr
    ) {
        
        // 创建一个新的节点
        std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(
            *steiner, steiner->fixedLayers, numDagNodes++
        );
        
        // 遍历steiner的子节点
        for (auto steinerChild : steiner->children) {
            // 递归调用函数，构建路由DAG
            constructDag(current, steinerChild);
        }
        // 如果dstNode为空，则将当前节点赋值给dstNode
        if (dstNode == nullptr) {
            dstNode = current;
        } else {
            // 否则将当前节点添加到dstNode的子节点中
            dstNode->children.emplace_back(current);
            // 构建路径
           
            constructPathsfixed_again(dstNode, current,ChoosedPatternIndexArray,NetInedex,this_tree_two_pin_net_index);
            //std::cout<<dstNode->x<<" "<<dstNode->y<<std::endl;
            //std::cout<<*dstNode.getPythonString(routingDag)<<std::endl;
        }
    };
    // 调用函数，构建路由DAG
    constructDag(routingDag, steinerTree);
    //outf<<routingDag->getPythonString(routingDag)<<std::endl;
    // outf<<steinerTree->getPythonString(steinerTree)<<std::endl;
    //outf.close();
    //std::cout<<routingDag->getPythonString(routingDag)<<std::endl;
    //std::cout<<steinerTree->x<<" "<<steinerTree->y<<std::endl;
    
}
/*
void PatternRoute::constructRoutingDAG() {
    // 1. Select access points
    robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>> selectedAccessPoints;
    // cell hash (2d) -> access point, fixed layer interval
    selectedAccessPoints.reserve(net.getNumPins());
    const auto& boundingBox = net.getBoundingBox();
    utils::PointT<int> netCenter(boundingBox.cx(), boundingBox.cy());
    for (const auto& accessPoints : net.getPinAccessPoints()) {
        std::pair<int, int> bestAccessDist = {0, std::numeric_limits<int>::max()};
        int bestIndex = -1;
        for (int index = 0; index < accessPoints.size(); index++) {
            const auto& point = accessPoints[index];
            int accessibility = 0;
            if (point.layerIdx >= parameters.min_routing_layer) {
                unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
                accessibility += gridGraph.getEdge(point.layerIdx, point.x, point.y).capacity >= 1;
                if (point[direction] > 0) {
                    auto lower = point;
                    lower[direction] -= 1;
                    accessibility += gridGraph.getEdge(lower.layerIdx, lower.x, lower.y).capacity >= 1;
                }
            } else {
                accessibility = 1;
            }
            int distance = abs(netCenter.x - point.x) + abs(netCenter.y - point.y);
            if (accessibility > bestAccessDist.first || (accessibility == bestAccessDist.first && distance < bestAccessDist.second)) {
                bestIndex = index;
                bestAccessDist = {accessibility, distance};
            }
        }
        if (bestAccessDist.first == 0) {
            log() << "Warning: the pin is hard to access." << std::endl; 
            for (auto pt : accessPoints) {
                // log() << gridGraph.getSize(0) << ", " << gridGraph.getSize(1) << std::endl;
                log() << gridGraph.getCellBox(pt).x.low / 2000.0 << " " << gridGraph.getCellBox(pt).y.low / 2000.0 << " " << gridGraph.getCellBox(pt).x.high / 2000.0 << " " << gridGraph.getCellBox(pt).y.high / 2000.0 << std::endl;
            }
        }
        const utils::PointT<int> selectedPoint = accessPoints[bestIndex];
        const uint64_t hash = gridGraph.hashCell(selectedPoint.x, selectedPoint.y);
        if (selectedAccessPoints.find(hash) == selectedAccessPoints.end()) {
            selectedAccessPoints.emplace(hash, std::make_pair(selectedPoint, utils::IntervalT<int>()));
        }
        utils::IntervalT<int>& fixedLayerInterval = selectedAccessPoints[hash].second;
        for (const auto& point : accessPoints) {
            if (point.x == selectedPoint.x && point.y == selectedPoint.y) {
                fixedLayerInterval.Update(point.layerIdx);
            }
        }
    }
    // Extend the fixed layers to 2 layers higher to facilitate track switching
    for (auto& accessPoint : selectedAccessPoints) {
        utils::IntervalT<int>& fixedLayers = accessPoint.second.second;
        fixedLayers.high = min(fixedLayers.high + 2, (int)gridGraph.getNumLayers() - 1);
    }
    // 2. Construct Steiner tree
    const int degree = selectedAccessPoints.size();
    if (degree == 1) {
        for (auto& accessPoint : selectedAccessPoints) {
            routingDag = std::make_shared<PatternRoutingNode>(accessPoint.second.first, accessPoint.second.second, numDagNodes++);
        }
    } else {
        int xs[degree * 4];
        int ys[degree * 4];
        int i = 0;
        for (auto& accessPoint : selectedAccessPoints) {
            xs[i] = accessPoint.second.first.x;
            ys[i] = accessPoint.second.first.y;
            i++;
        }
        Tree flutetree = flute(degree, xs, ys, ACCURACY);
        const int numBranches = degree + degree - 2;
        vector<utils::PointT<int>> steinerPoints;
        steinerPoints.reserve(numBranches);
        vector<vector<int>> adjacentList(numBranches);
        for (int branchIndex = 0; branchIndex < numBranches; branchIndex++) {
            const Branch& branch = flutetree.branch[branchIndex];
            steinerPoints.emplace_back(branch.x, branch.y);
            if (branchIndex == branch.n) continue;
            adjacentList[branchIndex].push_back(branch.n);
            adjacentList[branch.n].push_back(branchIndex);
        }
        std::function<void(std::shared_ptr<PatternRoutingNode>&, int, int)> constructNodes = 
        [&](std::shared_ptr<PatternRoutingNode>& parent, int prevIndex, int curIndex) {
            std::shared_ptr<PatternRoutingNode> current = std::make_shared<PatternRoutingNode>(steinerPoints[curIndex], numDagNodes++);
            if (parent != nullptr && parent->x == current->x && parent->y == current->y) {
                for (int nextIndex : adjacentList[curIndex]) {
                    if (nextIndex == prevIndex) continue;
                    constructNodes(parent, curIndex, nextIndex);
                }
                return;
            }
            // Build subtree
            for (int nextIndex : adjacentList[curIndex]) {
                if (nextIndex == prevIndex) continue;
                constructNodes(current, curIndex, nextIndex);
            }
            // Set fixed layer interval
            uint64_t hash = gridGraph.hashCell(current->x, current->y);
            if (selectedAccessPoints.find(hash) != selectedAccessPoints.end()) {
                current->fixedLayers = selectedAccessPoints[hash].second;
            }
            // Connect current to parent
            if (parent == nullptr) {
                parent = current;
            } else {
                parent->children.emplace_back(current);
                constructPaths(parent, current);
            }
        };
        int root = 0;
        // pick a root has a degree of 1
        std::function<bool(int)> hasDegree1 = [&] (int index) {
            if (adjacentList[index].size() == 1) {
                int nextIndex = adjacentList[index][0];
                if (steinerPoints[index] == steinerPoints[nextIndex]) {
                    return hasDegree1(nextIndex);
                } else {
                    return true;
                }
            } else {
                return false;
            }
        };
        for (int i = 0; i < steinerPoints.size(); i++) {
            if (hasDegree1(i)) {
                root = i;
                break;
            }
        }
        constructNodes(routingDag, -1, root);
    }
}
*/

void PatternRoute::constructPaths(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end, int childIndex) {
    if (childIndex == -1) {
        childIndex = start->paths.size();
        start->paths.emplace_back();
    }
    vector<std::shared_ptr<PatternRoutingNode>>& childPaths = start->paths[childIndex];
    if (start->x == end->x || start->y == end->y) {
        childPaths.push_back(end);
    } else {
        for (int pathIndex = 0; pathIndex <= 1; pathIndex++) { // two paths of different L-shape
            utils::PointT<int> midPoint = pathIndex ? utils::PointT<int>(start->x, end->y) : utils::PointT<int>(end->x, start->y);
            std::shared_ptr<PatternRoutingNode> mid = std::make_shared<PatternRoutingNode>(midPoint, numDagNodes++, true);
            mid->paths = {{end}};
            childPaths.push_back(mid);//由于是地址引用，所以这里赋值，就会导致start的paths也会被修改
        }
    }
}
// void PatternRoute::constructPaths(std::shared_ptr<GRTreeNode>& start, std::shared_ptr<GRTreeNode>& end, int childIndex) {
//     if (childIndex == -1) {
//         childIndex = start->paths.size();
//         start->paths.emplace_back();
//     }
//     vector<std::shared_ptr<PatternRoutingNode>>& childPaths = start->paths[childIndex];
//     if (start->x == end->x || start->y == end->y) {
//         childPaths.push_back(end);
//     } else {
//         for (int pathIndex = 0; pathIndex <= 1; pathIndex++) { // two paths of different L-shape
//             utils::PointT<int> midPoint = pathIndex ? utils::PointT<int>(start->x, end->y) : utils::PointT<int>(end->x, start->y);
//             std::shared_ptr<PatternRoutingNode> mid = std::make_shared<PatternRoutingNode>(midPoint, numDagNodes++, true);
//             mid->paths = {{end}};
//             childPaths.push_back(mid);
//         }
//     }
// }



void PatternRoute::constructPathsfixed(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,torch::Tensor& ChoosedPatternIndexArray,int& TwoPinNetInedex,int childIndex) {
    // 如果childIndex为-1，则将其设置为start->paths的最后一个元素的索引
    if (childIndex == -1) {
        childIndex = start->paths.size();
        start->paths.emplace_back();
    }
    //std::cout<<start->x<<" "<<start->y<<std::endl;
    // 获取start->paths中childIndex索引的路径
    vector<std::shared_ptr<PatternRoutingNode>>& childPaths = start->paths[childIndex];
    // 如果start和end的x坐标或y坐标相等，则将end添加到childPaths中
    if (start->x == end->x || start->y == end->y) {
        childPaths.push_back(end);
    } else {
        
        
        // 计算中间点
        auto mode = ChoosedPatternIndexArray[TwoPinNetInedex].item().to<int>();
        utils::PointT<int> midPoint = mode ? utils::PointT<int>(start->x, end->y) : utils::PointT<int>(end->x, start->y);
        // 创建一个PatternRoutingNode类型的指针，并设置其x和y坐标为中间点，numDagNodes自增，paths为一个元素，即end
        std::shared_ptr<PatternRoutingNode> mid = std::make_shared<PatternRoutingNode>(midPoint, numDagNodes++, true);
        mid->paths = {{end}};
        // 将mid添加到childPaths中
        childPaths.push_back(mid);
            
        
    }
    TwoPinNetInedex++;
}
void PatternRoute::constructPathsfixed55(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,int& ChoosedPatternIndexArray,int& TwoPinNetInedex,int childIndex) {
    // 如果childIndex为-1，则将其设置为start->paths的最后一个元素的索引
    //log()<<"start"<<start->x<<" "<<start->y<<std::endl;
    //log()<<"end"<<end->x<<" "<<end->y<<std::endl;
    //std::cout<<"TwoPinNetInedex"<<TwoPinNetInedex<<std::endl;
    if (childIndex == -1) {
        childIndex = start->paths.size();
        start->paths.emplace_back();
    }
    //std::cout<<start->x<<" "<<start->y<<std::endl;
    // 获取start->paths中childIndex索引的路径
    vector<std::shared_ptr<PatternRoutingNode>>& childPaths = start->paths[childIndex];
    // 如果start和end的x坐标或y坐标相等，则将end添加到childPaths中
    if (start->x == end->x && start->y == end->y) {
        childPaths.push_back(end);
        TwoPinNetInedex++;
    }
    else if (start->x == end->x || start->y == end->y) {
        childPaths.push_back(end);
        TwoPinNetInedex++;
    } else {
        
        
        // 计算中间点
        // auto mode = ChoosedPatternIndexArray[TwoPinNetInedex].item().to<int>();
        auto mode = ChoosedPatternIndexArray;
        utils::PointT<int> midPoint = mode ? utils::PointT<int>(start->x, end->y) : utils::PointT<int>(end->x, start->y);
        // 创建一个PatternRoutingNode类型的指针，并设置其x和y坐标为中间点，numDagNodes自增，paths为一个元素，即end
        std::shared_ptr<PatternRoutingNode> mid = std::make_shared<PatternRoutingNode>(midPoint, numDagNodes++, true);
        mid->paths = {{end}};
        // 将mid添加到childPaths中
        childPaths.push_back(mid);
            
        TwoPinNetInedex++;
    }
    
    //log()<<"TwoPinNetInedex:"<<TwoPinNetInedex<<std::endl;
}
void PatternRoute::constructPathsfixed_again(std::shared_ptr<PatternRoutingNode>& start, std::shared_ptr<PatternRoutingNode>& end,torch::Tensor& ChoosedPatternIndexArray,int NetIndex,int& TwoPinNetInedex,int childIndex) {
    // 如果childIndex为-1，则将其设置为start->paths的最后一个元素的索引
    if (childIndex == -1) {
        childIndex = start->paths.size();
        start->paths.emplace_back();
    }
    //std::cout<<start->x<<" "<<start->y<<std::endl;
    // 获取start->paths中childIndex索引的路径
    vector<std::shared_ptr<PatternRoutingNode>>& childPaths = start->paths[childIndex];
    // 如果start和end的x坐标或y坐标相等，则将end添加到childPaths中
    if (start->x == end->x || start->y == end->y) {
        childPaths.push_back(end);
    } else {
        
        
        // 计算中间点
        auto mode = ChoosedPatternIndexArray[NetIndex][0][TwoPinNetInedex].item().to<int>();
        utils::PointT<int> midPoint = mode ? utils::PointT<int>(start->x, end->y) : utils::PointT<int>(end->x, start->y);
        // 创建一个PatternRoutingNode类型的指针，并设置其x和y坐标为中间点，numDagNodes自增，paths为一个元素，即end
        std::shared_ptr<PatternRoutingNode> mid = std::make_shared<PatternRoutingNode>(midPoint, numDagNodes++, true);
        mid->paths = {{end}};
        // 将mid添加到childPaths中
        childPaths.push_back(mid);
            
        
    }
    TwoPinNetInedex++;
}


void PatternRoute::constructDetours(GridGraphView<bool>& congestionView) {
    struct ScaffoldNode {
        std::shared_ptr<PatternRoutingNode> node;
        vector<std::shared_ptr<ScaffoldNode>> children;
        ScaffoldNode(std::shared_ptr<PatternRoutingNode> n): node(n) {}
    };
    
    vector<vector<std::shared_ptr<ScaffoldNode>>> scaffolds(2);
    vector<vector<std::shared_ptr<ScaffoldNode>>> scaffoldNodes(
        2, vector<std::shared_ptr<ScaffoldNode>>(numDagNodes, nullptr)
    ); // direction -> numDagNodes -> scaffold node
    vector<bool> visited(numDagNodes, false);
    
    std::function<void(std::shared_ptr<PatternRoutingNode>)> buildScaffolds = 
        [&] (std::shared_ptr<PatternRoutingNode> node) {
            if (visited[node->index]) return;
            visited[node->index] = true;
            
            if (node->optional) {
                assert(node->paths.size() == 1 && node->paths[0].size() == 1 && !node->paths[0][0]->optional);
                auto& path = node->paths[0][0];
                buildScaffolds(path);
                unsigned direction = (node->y == path->y ? MetalLayer::H : MetalLayer::V);
                if (!scaffoldNodes[direction][path->index] && congestionView.check(*node, *path)) {
                    scaffoldNodes[direction][path->index] = std::make_shared<ScaffoldNode>(path);
                }
            } else {
                for (auto& childPaths : node->paths) {
                    for (auto& path : childPaths) {
                        buildScaffolds(path);
                        unsigned direction = (node->y == path->y ? MetalLayer::H : MetalLayer::V);
                        if (path->optional) {
                            if (!scaffoldNodes[direction][node->index] && congestionView.check(*node, *path)) {
                                scaffoldNodes[direction][node->index] = std::make_shared<ScaffoldNode>(node);
                            }
                        } else {
                            if (congestionView.check(*node, *path)) {
                                if (!scaffoldNodes[direction][node->index]) {
                                    scaffoldNodes[direction][node->index] = std::make_shared<ScaffoldNode>(node);
                                }
                                if (!scaffoldNodes[direction][path->index]) {
                                    scaffoldNodes[direction][node->index]->children.emplace_back(std::make_shared<ScaffoldNode>(path));
                                } else {
                                    scaffoldNodes[direction][node->index]->children.emplace_back(scaffoldNodes[direction][path->index]);
                                    scaffoldNodes[direction][path->index] = nullptr;
                                }
                            }
                        }
                    }
                    for (auto& child : node->children) {
                        for (unsigned direction = 0; direction < 2; direction++) {
                            if (scaffoldNodes[direction][child->index]) {
                                scaffolds[direction].emplace_back(std::make_shared<ScaffoldNode>(node));
                                scaffolds[direction].back()->children.emplace_back(scaffoldNodes[direction][child->index]);
                                scaffoldNodes[direction][child->index] = nullptr;
                            }
                        }
                    }
                }
            }
        };
        
    buildScaffolds(routingDag);
    for (unsigned direction = 0; direction < 2; direction++) {
        if (scaffoldNodes[direction][routingDag->index]) {
            scaffolds[direction].emplace_back(std::make_shared<ScaffoldNode>(nullptr));
            scaffolds[direction].back()->children.emplace_back(scaffoldNodes[direction][routingDag->index]);
        }
    }
    
    
    // std::function<void(std::shared_ptr<ScaffoldNode>)> printScaffold = 
    //     [&] (std::shared_ptr<ScaffoldNode> scaffoldNode) {
    // 
    //         for (auto& scaffoldChild : scaffoldNode->children) {
    //             if (scaffoldNode->node) std::cout << (utils::PointT<int>)*scaffoldNode->node << " " << scaffoldNode->children.size();
    //             else std::cout << "null";
    //             std::cout << " -> " << (utils::PointT<int>)*scaffoldChild->node;
    //             if (scaffoldNode->node) std::cout << " " << congested(scaffoldNode->node, scaffoldChild->node);
    //             std::cout << std::endl;
    //             printScaffold(scaffoldChild);
    //         }
    //     };
    // 
    // for (unsigned direction = 0; direction < 1; direction++) {
    //     for (auto scaffold : scaffolds[direction]) {
    //         printScaffold(scaffold);
    //         std::cout << std::endl;
    //     }
    // }
    
    std::function<void(std::shared_ptr<ScaffoldNode>, utils::IntervalT<int>&, vector<int>&, unsigned, bool)> getTrunkAndStems = 
        [&] (std::shared_ptr<ScaffoldNode> scaffoldNode, utils::IntervalT<int>& trunk, vector<int>& stems, unsigned direction, bool starting) {
            if (starting) {
                if (scaffoldNode->node) {
                    stems.emplace_back((*scaffoldNode->node)[1 - direction]);
                    trunk.Update((*scaffoldNode->node)[direction]);
                }
                for (auto& scaffoldChild : scaffoldNode->children) getTrunkAndStems(scaffoldChild, trunk, stems, direction, false);
            } else {
                trunk.Update((*scaffoldNode->node)[direction]);
                if (scaffoldNode->node->fixedLayers.IsValid()) {
                    stems.emplace_back((*scaffoldNode->node)[1 - direction]);
                }
                for (auto& treeChild : scaffoldNode->node->children) {
                    bool scaffolded = false;
                    for (auto& scaffoldChild : scaffoldNode->children) {
                        if (treeChild == scaffoldChild->node) {
                            getTrunkAndStems(scaffoldChild, trunk, stems, direction, false);
                            scaffolded = true;
                            break;
                        }
                    }
                    if (!scaffolded) {
                        stems.emplace_back((*treeChild)[1 - direction]);
                        trunk.Update((*treeChild)[direction]);
                    }
                }
            }
        };
    
    auto getTotalStemLength = [&] (const vector<int>& stems, const int pos) {
        int length = 0;
        for (int stem : stems) length += abs(stem - pos);
        return length;
    };
    
    std::function<std::shared_ptr<PatternRoutingNode>(std::shared_ptr<ScaffoldNode>, unsigned, int)> buildDetour = 
        [&] (std::shared_ptr<ScaffoldNode> scaffoldNode, unsigned direction, int shiftAmount) {
            std::shared_ptr<PatternRoutingNode> treeNode = scaffoldNode->node;
            if (treeNode->fixedLayers.IsValid()) {
                std::shared_ptr<PatternRoutingNode> dupTreeNode = 
                    std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, treeNode->fixedLayers, numDagNodes++);
                std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
                    std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
                (*shiftedTreeNode)[1 - direction] += shiftAmount;
                constructPaths(shiftedTreeNode, dupTreeNode);
                for (auto& treeChild : treeNode->children) {
                    bool built = false;
                    for (auto& scaffoldChild : scaffoldNode->children) {
                        if (treeChild == scaffoldChild->node) {
                            auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
                            constructPaths(shiftedTreeNode, shiftedChildTreeNode);
                            built = true;
                            break;
                        }
                    }
                    if (!built) {
                        constructPaths(shiftedTreeNode, treeChild);
                    }
                }
                return shiftedTreeNode;
            } else {
                std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
                    std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
                (*shiftedTreeNode)[1 - direction] += shiftAmount;
                for (auto& treeChild : treeNode->children) {
                    bool built = false;
                    for (auto& scaffoldChild : scaffoldNode->children) {
                        if (treeChild == scaffoldChild->node) {
                            auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
                            constructPaths(shiftedTreeNode, shiftedChildTreeNode);
                            built = true;
                            break;
                        }
                    }
                    if (!built) {
                        constructPaths(shiftedTreeNode, treeChild);
                    }
                }
                return shiftedTreeNode;
            }
        };
        
    for (unsigned direction = 0; direction < 2; direction++) {
        for (std::shared_ptr<ScaffoldNode> scaffold : scaffolds[direction]) {
            assert (scaffold->children.size() == 1);
            
            utils::IntervalT<int> trunk;
            vector<int> stems;
            getTrunkAndStems(scaffold, trunk, stems, direction, true);
            std::sort(stems.begin(), stems.end());
            int trunkPos = (*scaffold->children[0]->node)[1 - direction];
            int originalLength = getTotalStemLength(stems, trunkPos);
            utils::IntervalT<int> shiftInterval(trunkPos);
            int maxLengthIncrease = trunk.range() * parameters.max_detour_ratio;
            while (shiftInterval.low - 1 >= 0 && getTotalStemLength(stems, shiftInterval.low - 1) - originalLength <= maxLengthIncrease) shiftInterval.low--;
            while (shiftInterval.high + 1 < gridGraph.getSize(1 - direction) && getTotalStemLength(stems, shiftInterval.high - 1) - originalLength <= maxLengthIncrease) shiftInterval.high++;
            int step = 1;
            while ((trunkPos - shiftInterval.low) / (step + 1) + (shiftInterval.high - trunkPos) / (step + 1) >= parameters.target_detour_count) step++;
            utils::IntervalT<int> dupShiftInterval = shiftInterval;
            shiftInterval.low = trunkPos - (trunkPos - shiftInterval.low) / step * step;
            shiftInterval.high = trunkPos + (shiftInterval.high - trunkPos) / step * step;
            for (double pos = shiftInterval.low; pos <= shiftInterval.high; pos += step) {
                int shiftAmount = (pos - trunkPos); 
                if (shiftAmount == 0) continue;
                if (scaffold->node) {
                    auto& scaffoldChild = scaffold->children[0];
                    if ((*scaffoldChild->node)[1 - direction] + shiftAmount < 0 || 
                        (*scaffoldChild->node)[1 - direction] + shiftAmount >= gridGraph.getSize(1 - direction)) {
                        continue;
                    }
                    for (int childIndex = 0; childIndex < scaffold->node->children.size(); childIndex++) {
                        auto& treeChild = scaffold->node->children[childIndex];
                        if (treeChild == scaffoldChild->node) {
                            std::shared_ptr<PatternRoutingNode> shiftedChild = buildDetour(scaffoldChild, direction, shiftAmount);
                            constructPaths(scaffold->node, shiftedChild, childIndex);
                        }
                    }
                } else {
                    std::shared_ptr<ScaffoldNode> scaffoldNode = scaffold->children[0];
                    auto treeNode = scaffoldNode->node;
                    if (treeNode->children.size() == 1) {
                        if ((*treeNode)[1 - direction] + shiftAmount < 0 || 
                            (*treeNode)[1 - direction] + shiftAmount >= gridGraph.getSize(1 - direction)) {
                            continue;
                        }
                        std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
                            std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
                        (*shiftedTreeNode)[1 - direction] += shiftAmount;
                        constructPaths(treeNode, shiftedTreeNode, 0);
                        for (auto& treeChild : treeNode->children) {
                            bool built = false;
                            for (auto& scaffoldChild : scaffoldNode->children) {
                                if (treeChild == scaffoldChild->node) {
                                    auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
                                    constructPaths(shiftedTreeNode, shiftedChildTreeNode);
                                    built = true;
                                    break;
                                }
                            }
                            if (!built) {
                                constructPaths(shiftedTreeNode, treeChild);
                            }
                        }
                    
                    } else {
                        log() << "Warning: the root has not exactly one child." << std::endl;
                    }
                }
            }
        }
    }
}

// void PatternRoute::constructDetours(GridGraphView<bool>& congestionView) {
//     // 定义支架节点结构体
//     struct ScaffoldNode {
//         std::shared_ptr<PatternRoutingNode> node; // 节点指针
//         vector<std::shared_ptr<ScaffoldNode>> children; // 子节点指针数组
//         ScaffoldNode(std::shared_ptr<PatternRoutingNode> n): node(n) {} // 构造函数
//     };
    
//     // 初始化变量
//     vector<vector<std::shared_ptr<ScaffoldNode>>> scaffolds(2); // 支架数组
//     vector<vector<std::shared_ptr<ScaffoldNode>>> scaffoldNodes( // 节点数组
//         2, vector<std::shared_ptr<ScaffoldNode>>(numDagNodes, nullptr)
//     );
//     vector<bool> visited(numDagNodes, false); // 标记数组
    
//     // 构建支架节点的递归函数
//     std::function<void(std::shared_ptr<PatternRoutingNode>)> buildScaffolds = 
//         [&] (std::shared_ptr<PatternRoutingNode> node) {
//             if (visited[node->index]) return; // 如果节点已访问过，则返回
//             visited[node->index] = true; // 标记节点为已访问
            
//             // 如果节点是可选的
//             if (node->optional) {
//                 assert(node->paths.size() == 1 && node->paths[0].size() == 1 && !node->paths[0][0]->optional); // 断言路径
//                 auto& path = node->paths[0][0];
//                 buildScaffolds(path); // 递归构建路径支架
//                 unsigned direction = (node->y == path->y ? MetalLayer::H : MetalLayer::V); // 确定方向
//                 // 如果支架节点为空且不拥塞，则创建支架节点
//                 if (!scaffoldNodes[direction][path->index] && congestionView.check(*node, *path)) {
//                     scaffoldNodes[direction][path->index] = std::make_shared<ScaffoldNode>(path);
//                 }
//             } else {
//                 // 遍历节点的路径
//                 for (auto& childPaths : node->paths) {
//                     for (auto& path : childPaths) {
//                         buildScaffolds(path); // 递归构建路径支架
//                         unsigned direction = (node->y == path->y ? MetalLayer::H : MetalLayer::V); // 确定方向
//                         // 如果路径节点是可选的
//                         if (path->optional) {
//                             // 如果支架节点为空且不拥塞，则创建支架节点
//                             if (!scaffoldNodes[direction][node->index] && congestionView.check(*node, *path)) {
//                                 scaffoldNodes[direction][node->index] = std::make_shared<ScaffoldNode>(node);
//                             }
//                         } else {
//                             // 如果路径节点不可选且不拥塞
//                             if (congestionView.check(*node, *path)) {
//                                 // 如果支架节点为空，则创建支架节点
//                                 if (!scaffoldNodes[direction][node->index]) {
//                                     scaffoldNodes[direction][node->index] = std::make_shared<ScaffoldNode>(node);
//                                 }
//                                 // 如果路径节点为空，则创建路径节点，并添加到支架节点的子节点数组中
//                                 if (!scaffoldNodes[direction][path->index]) {
//                                     scaffoldNodes[direction][node->index]->children.emplace_back(std::make_shared<ScaffoldNode>(path));
//                                 } else {
//                                     // 如果路径节点已存在，则将其添加到支架节点的子节点数组中，并清空路径节点
//                                     scaffoldNodes[direction][node->index]->children.emplace_back(scaffoldNodes[direction][path->index]);
//                                     scaffoldNodes[direction][path->index] = nullptr;
//                                 }
//                             }
//                         }
//                     }
//                     // 遍历节点的子节点，如果子节点的支架节点存在，则将其添加到支架数组中
//                     for (auto& child : node->children) {
//                         for (unsigned direction = 0; direction < 2; direction++) {
//                             if (scaffoldNodes[direction][child->index]) {
//                                 scaffolds[direction].emplace_back(std::make_shared<ScaffoldNode>(node));
//                                 scaffolds[direction].back()->children.emplace_back(scaffoldNodes[direction][child->index]);
//                                 scaffoldNodes[direction][child->index] = nullptr;
//                             }
//                         }
//                     }
//                 }
//             }
//         };
    
//     // 调用递归函数构建支架节点
//     buildScaffolds(routingDag);
//     for (unsigned direction = 0; direction < 2; direction++) {
//         // 如果根节点的支架节点存在，则将其添加到支架数组中
//         if (scaffoldNodes[direction][routingDag->index]) {
//             scaffolds[direction].emplace_back(std::make_shared<ScaffoldNode>(nullptr));
//             scaffolds[direction].back()->children.emplace_back(scaffoldNodes[direction][routingDag->index]);
//         }
//     }
    
//     // 定义获取主干和茎的递归函数
//     std::function<void(std::shared_ptr<ScaffoldNode>, utils::IntervalT<int>&, vector<int>&, unsigned, bool)> getTrunkAndStems = 
//         [&] (std::shared_ptr<ScaffoldNode> scaffoldNode, utils::IntervalT<int>& trunk, vector<int>& stems, unsigned direction, bool starting) {
//             if (starting) {
//                 // 如果是起始节点，则获取主干和茎
//                 if (scaffoldNode->node) {
//                     stems.emplace_back((*scaffoldNode->node)[1 - direction]);
//                     trunk.Update((*scaffoldNode->node)[direction]);
//                 }
//                 // 递归处理子节点
//                 for (auto& scaffoldChild : scaffoldNode->children) getTrunkAndStems(scaffoldChild, trunk, stems, direction, false);
//             } else {
//                 // 如果不是起始节点，则继续获取主干和茎
//                 trunk.Update((*scaffoldNode->node)[direction]);
//                 if (scaffoldNode->node->fixedLayers.IsValid()) {
//                     stems.emplace_back((*scaffoldNode->node)[1 - direction]);
//                 }
//                 // 递归处理子节点
//                 for (auto& treeChild : scaffoldNode->node->children) {
//                     bool scaffolded = false;
//                     for (auto& scaffoldChild : scaffoldNode->children) {
//                         if (treeChild == scaffoldChild->node) {
//                             getTrunkAndStems(scaffoldChild, trunk, stems, direction, false);
//                             scaffolded = true;
//                             break;
//                         }
//                     }
//                     if (!scaffolded) {
//                         stems.emplace_back((*treeChild)[1 - direction]);
//                         trunk.Update((*treeChild)[direction]);
//                     }
//                 }
//             }
//         };
    
//     // 计算总茎长的Lambda函数
//     auto getTotalStemLength = [&] (const vector<int>& stems, const int pos) {
//         int length = 0;
//         for (int stem : stems) length += abs(stem - pos);
//         return length;
//     };
    
//     // 定义构建侧路的递归函数
//     std::function<std::shared_ptr<PatternRoutingNode>(std::shared_ptr<ScaffoldNode>, unsigned, int)> buildDetour = 
//         [&] (std::shared_ptr<ScaffoldNode> scaffoldNode, unsigned direction, int shiftAmount) {
//             std::shared_ptr<PatternRoutingNode> treeNode = scaffoldNode->node;
//             // 如果节点有固定层，则创建一个偏移后的节点，并构建侧路
//             if (treeNode->fixedLayers.IsValid()) {
//                 std::shared_ptr<PatternRoutingNode> dupTreeNode = 
//                     std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, treeNode->fixedLayers, numDagNodes++);
//                 std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
//                     std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
//                 (*shiftedTreeNode)[1 - direction] += shiftAmount;
//                 constructPaths(shiftedTreeNode, dupTreeNode);
//                 for (auto& treeChild : treeNode->children) {
//                     bool built = false;
//                     for (auto& scaffoldChild : scaffoldNode->children) {
//                         if (treeChild == scaffoldChild->node) {
//                             auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
//                             constructPaths(shiftedTreeNode, shiftedChildTreeNode);
//                             built = true;
//                             break;
//                         }
//                     }
//                     if (!built) {
//                         constructPaths(shiftedTreeNode, treeChild);
//                     }
//                 }
//                 return shiftedTreeNode;
//             } else {
//                 // 如果节点没有固定层，则创建一个偏移后的节点，并构建侧路
//                 std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
//                     std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
//                 (*shiftedTreeNode)[1 - direction] += shiftAmount;
//                 for (auto& treeChild : treeNode->children) {
//                     bool built = false;
//                     for (auto& scaffoldChild : scaffoldNode->children) {
//                         if (treeChild == scaffoldChild->node) {
//                             auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
//                             constructPaths(shiftedTreeNode, shiftedChildTreeNode);
//                             built = true;
//                             break;
//                         }
//                     }
//                     if (!built) {
//                         constructPaths(shiftedTreeNode, treeChild);
//                     }
//                 }
//                 return shiftedTreeNode;
//             }
//         };
        
//     // 遍历支架数组，处理每个支架节点
//     for (unsigned direction = 0; direction < 2; direction++) {
//         for (std::shared_ptr<ScaffoldNode> scaffold : scaffolds[direction]) {
//             assert (scaffold->children.size() == 1); // 断言支架节点的子节点数量为1
            
//             utils::IntervalT<int> trunk;
//             vector<int> stems;
//             // 获取支架节点的主干和茎
//             getTrunkAndStems(scaffold, trunk, stems, direction, true);
//             std::sort(stems.begin(), stems.end()); // 对茎进行排序
//             int trunkPos = (*scaffold->children[0]->node)[1 - direction]; // 主干位置
//             int originalLength = getTotalStemLength(stems, trunkPos); // 原始茎长
//             utils::IntervalT<int> shiftInterval(trunkPos); // 偏移范围
//             int maxLengthIncrease = trunk.range() * parameters.max_detour_ratio; // 最大增加茎长
//             while (shiftInterval.low - 1 >= 0 && getTotalStemLength(stems, shiftInterval.low - 1) - originalLength <= maxLengthIncrease) shiftInterval.low--; // 扩展偏移范围
//             while (shiftInterval.high + 1 < gridGraph.getSize(1 - direction) && getTotalStemLength(stems, shiftInterval.high - 1) - originalLength <= maxLengthIncrease) shiftInterval.high++;
//             int step = 1;
//             // 计算步长
//             while ((trunkPos - shiftInterval.low) / (step + 1) + (shiftInterval.high - trunkPos) / (step + 1) >= parameters.target_detour_count) step++;
//             utils::IntervalT<int> dupShiftInterval = shiftInterval;
//             shiftInterval.low = trunkPos - (trunkPos - shiftInterval.low) / step * step;
//             shiftInterval.high = trunkPos + (shiftInterval.high - trunkPos) / step * step;
//             // 在偏移范围内构建侧路
//             for (double pos = shiftInterval.low; pos <= shiftInterval.high; pos += step) {
//                 int shiftAmount = (pos - trunkPos); 
//                 if (shiftAmount == 0) continue; // 如果偏移量为0，则跳过
//                 // 如果支架节点存在，则创建偏移后的侧路节点
//                 if (scaffold->node) {
//                     auto& scaffoldChild = scaffold->children[0];
//                     if ((*scaffoldChild->node)[1 - direction] + shiftAmount < 0 || 
//                         (*scaffoldChild->node)[1 - direction] + shiftAmount >= gridGraph.getSize(1 - direction)) {
//                         continue;
//                     }
//                     for (int childIndex = 0; childIndex < scaffold->node->children.size(); childIndex++) {
//                         auto& treeChild = scaffold->node->children[childIndex];
//                         if (treeChild == scaffoldChild->node) {
//                             std::shared_ptr<PatternRoutingNode> shiftedChild = buildDetour(scaffoldChild, direction, shiftAmount);
//                             constructPaths(scaffold->node, shiftedChild, childIndex);
//                         }
//                     }
//                 } else {
//                     // 如果支架节点不存在，则创建偏移后的侧路节点
//                     std::shared_ptr<ScaffoldNode> scaffoldNode = scaffold->children[0];
//                     auto treeNode = scaffoldNode->node;
//                     if (treeNode->children.size() == 1) {
//                         if ((*treeNode)[1 - direction] + shiftAmount < 0 || 
//                             (*treeNode)[1 - direction] + shiftAmount >= gridGraph.getSize(1 - direction)) {
//                             continue;
//                         }
//                         std::shared_ptr<PatternRoutingNode> shiftedTreeNode = 
//                             std::make_shared<PatternRoutingNode>((utils::PointT<int>)*treeNode, numDagNodes++);
//                         (*shiftedTreeNode)[1 - direction] += shiftAmount;
//                         constructPaths(treeNode, shiftedTreeNode, 0);
//                         for (auto& treeChild : treeNode->children) {
//                             bool built = false;
//                             for (auto& scaffoldChild : scaffoldNode->children) {
//                                 if (treeChild == scaffoldChild->node) {
//                                     auto shiftedChildTreeNode = buildDetour(scaffoldChild, direction, shiftAmount);
//                                     constructPaths(shiftedTreeNode, shiftedChildTreeNode);
//                                     built = true;
//                                     break;
//                                 }
//                             }
//                             if (!built) {
//                                 constructPaths(shiftedTreeNode, treeChild);
//                             }
//                         }
                    
//                     } else {
//                         log() << "Warning: the root has not exactly one child." << std::endl;
//                     }
//                 }
//             }
//         }
//     }
// }

void PatternRoute::run() {
    //net.clearRoutingTree();
    calculateRoutingCosts(routingDag);
    //std::cout<<"done"<<std::endl;
    net.setRoutingTree(getRoutingTree(routingDag));
    //std::cout<<"done"<<std::endl;
}

void PatternRoute::calculateRoutingCosts(std::shared_ptr<PatternRoutingNode>& node) {
    if (node->costs.size() != 0) return;
    vector<vector<std::pair<CostT, int>>> childCosts; // childIndex -> layerIndex -> (cost, pathIndex)
    // Calculate child costs
    if (node->paths.size() > 0) childCosts.resize(node->paths.size());
    for (int childIndex = 0; childIndex < node->paths.size(); childIndex++) {
        auto& childPaths = node->paths[childIndex];
        auto& costs = childCosts[childIndex];
        costs.assign(gridGraph.getNumLayers(), {std::numeric_limits<CostT>::max(), -1});
        for (int pathIndex = 0; pathIndex < childPaths.size(); pathIndex++) {
            std::shared_ptr<PatternRoutingNode>& path = childPaths[pathIndex];
            calculateRoutingCosts(path);
            unsigned direction = node->x == path->x ? MetalLayer::V : MetalLayer::H;
            assert((*node)[1 - direction] == (*path)[1 - direction]);
            for (int layerIndex = parameters.min_routing_layer; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
                if (gridGraph.getLayerDirection(layerIndex) != direction) continue;
                CostT cost = path->costs[layerIndex] + gridGraph.getWireCost(layerIndex, *node, *path);
                if (cost < costs[layerIndex].first) costs[layerIndex] = std::make_pair(cost, pathIndex);
            }
        }
    }
    
    node->costs.assign(gridGraph.getNumLayers(), std::numeric_limits<CostT>::max());
    node->bestPaths.resize(gridGraph.getNumLayers());
    if (node->paths.size() > 0) {
        for (int layerIndex = 1; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
            node->bestPaths[layerIndex].assign(node->paths.size(), {-1, -1});
        }
    }
    // Calculate the partial sum of the via costs
    vector<CostT> viaCosts(gridGraph.getNumLayers());
    viaCosts[0] = 0;
    for (int layerIndex = 1; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
        viaCosts[layerIndex] = viaCosts[layerIndex - 1] + gridGraph.getViaCost(layerIndex - 1, *node);
    }
    utils::IntervalT<int> fixedLayers = node->fixedLayers;
    fixedLayers.low = min(fixedLayers.low, static_cast<int>(gridGraph.getNumLayers()) - 1);
    fixedLayers.high = max(fixedLayers.high, parameters.min_routing_layer);
    
    for (int lowLayerIndex = 0; lowLayerIndex <= fixedLayers.low; lowLayerIndex++) {
        vector<CostT> minChildCosts; 
        vector<std::pair<int, int>> bestPaths; 
        if (node->paths.size() > 0) {
            minChildCosts.assign(node->paths.size(), std::numeric_limits<CostT>::max());
            bestPaths.assign(node->paths.size(), {-1, -1});
        }
        for (int layerIndex = lowLayerIndex; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
            for (int childIndex = 0; childIndex < node->paths.size(); childIndex++) {
                if (childCosts[childIndex][layerIndex].first < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = childCosts[childIndex][layerIndex].first;
                    bestPaths[childIndex] = std::make_pair(childCosts[childIndex][layerIndex].second, layerIndex);
                }
            }
            if (layerIndex >= fixedLayers.high) {
                CostT cost = viaCosts[layerIndex] - viaCosts[lowLayerIndex];
                for (CostT childCost : minChildCosts) cost += childCost;
                if (cost < node->costs[layerIndex]) {
                    node->costs[layerIndex] = cost;
                    node->bestPaths[layerIndex] = bestPaths;
                }
            }
        }
        for (int layerIndex = gridGraph.getNumLayers() - 2; layerIndex >= lowLayerIndex; layerIndex--) {
            if (node->costs[layerIndex + 1] < node->costs[layerIndex]) {
                node->costs[layerIndex] = node->costs[layerIndex + 1];
                node->bestPaths[layerIndex] = node->bestPaths[layerIndex + 1];
            }
        }
    }
}

std::shared_ptr<GRTreeNode> PatternRoute::getRoutingTree(std::shared_ptr<PatternRoutingNode>& node, int parentLayerIndex) {
    //std::cout<<"this node location"<<node->x<<"~"<<node->y<<std::endl;
    if (parentLayerIndex == -1) {
        CostT minCost = std::numeric_limits<CostT>::max();
        for (int layerIndex = 0; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
            if (routingDag->costs[layerIndex] < minCost) {
                minCost = routingDag->costs[layerIndex];
                parentLayerIndex = layerIndex;
            }
        }
    }
    
    std::shared_ptr<GRTreeNode> routingNode = std::make_shared<GRTreeNode>(parentLayerIndex, node->x, node->y);
    std::shared_ptr<GRTreeNode> lowestRoutingNode = routingNode;
    std::shared_ptr<GRTreeNode> highestRoutingNode = routingNode;
    
    if (node->paths.size() > 0) {
        int pathIndex, layerIndex;
        vector<vector<std::shared_ptr<PatternRoutingNode>>> pathsOnLayer(gridGraph.getNumLayers());
        
        for (int childIndex = 0; childIndex < node->paths.size(); childIndex++) {
            //std::cout<<"thislineisok"<<std::endl;
            std::tie(pathIndex, layerIndex) = node->bestPaths[parentLayerIndex][childIndex];
            //std::cout<<"thislineisok"<<std::endl;
            pathsOnLayer[layerIndex].push_back(node->paths[childIndex][pathIndex]);
        }
        
        if (pathsOnLayer[parentLayerIndex].size() > 0) {
            for (auto& path : pathsOnLayer[parentLayerIndex]) {
                routingNode->children.push_back(getRoutingTree(path, parentLayerIndex));
            }
        }
        for (int layerIndex = parentLayerIndex - 1; layerIndex >= 0; layerIndex--) {
            if (pathsOnLayer[layerIndex].size() > 0) {
                lowestRoutingNode->children.push_back(std::make_shared<GRTreeNode>(layerIndex, node->x, node->y));
                lowestRoutingNode = lowestRoutingNode->children.back();
                for (auto& path : pathsOnLayer[layerIndex]) {
                    lowestRoutingNode->children.push_back(getRoutingTree(path, layerIndex));
                }
            }
        }
        for (int layerIndex = parentLayerIndex + 1; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
            if (pathsOnLayer[layerIndex].size() > 0) {
                highestRoutingNode->children.push_back(std::make_shared<GRTreeNode>(layerIndex, node->x, node->y));
                highestRoutingNode = highestRoutingNode->children.back();
                for (auto& path : pathsOnLayer[layerIndex]) {
                    highestRoutingNode->children.push_back(getRoutingTree(path, layerIndex));
                }
            }
        }
    }
    
    if (lowestRoutingNode->layerIdx > node->fixedLayers.low) {
        lowestRoutingNode->children.push_back(std::make_shared<GRTreeNode>(node->fixedLayers.low, node->x, node->y));
    }
    if (highestRoutingNode->layerIdx < node->fixedLayers.high) {
        highestRoutingNode->children.push_back(std::make_shared<GRTreeNode>(node->fixedLayers.high, node->x, node->y));
    }
    return routingNode;
}
