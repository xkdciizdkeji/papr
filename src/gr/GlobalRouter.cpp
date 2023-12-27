#include "GlobalRouter.h"
#include "PatternRoute.h"
#include "MazeRoute.h"
#include <chrono>
#include <random>
#include <algorithm> // 包含shuffle函数所需的头文件

// GlobalRouter::GlobalRouter(const Design& design, const Parameters& params): 
//     gridGraph(design, params), parameters(params) {
//     // Instantiate the global routing netlist
//     const vector<Net>& baseNets = design.getAllNets();
//     nets.reserve(baseNets.size());
//     for (const Net& baseNet : baseNets) {
//         nets.emplace_back(baseNet, design, gridGraph);
//     }
// }

GlobalRouter::GlobalRouter(const Parser &parser, const ParametersISPD24 &params):
    gridGraph(parser, params), parameters(params) {
    // Instantiate the global routing netlist
    const vector<PNet>& baseNets = parser.getAllNets();
    nets.reserve(baseNets.size());
    for (const PNet& baseNet : baseNets) {
        // GRNet grNet(baseNet);
        nets.emplace_back(baseNet);
    }
    // for(const auto& net:nets){
    //     log() << "net: " << net.getIndex() << " " << net.getName() << std::endl;
    //     for(const auto& accessPoints: net.getPinAccessPoints()) {
    //         log() << "points num: "<< accessPoints.size()<<std::endl;
    //         for(const auto& point: accessPoints) {
    //             log() << "point: " << point.layerIdx << " " << point.x << " " << point.y << std::endl;
    //         }
    //     }
    // }
    numofThreads = params.threads;
}


void GlobalRouter::route()
{
    int n1 = 0, n2 = 0, n3 = 0;
    double t1 = 0, t2 = 0, t3 = 0;
    
    auto t = std::chrono::high_resolution_clock::now();
    
    vector<int> netIndices;
    vector<int> netOverflows(nets.size());
    // vector<int> netScores(nets.size());
    netIndices.reserve(nets.size());
    for (const auto& net : nets) netIndices.push_back(net.getIndex());
    // Stage 1: Pattern routing
    n1 = netIndices.size();
    PatternRoute::readFluteLUT();
    log() << "stage 1: pattern routing" << std::endl;
    sortNetIndices(netIndices);
    // sortNetIndicesD(netIndices);
    // sortNetIndicesOLD(netIndices);
    // sortNetIndicesOLI(netIndices);

    vector<SingleNetRouter> routers;
    routers.reserve(netIndices.size());
    for (auto id : netIndices) routers.emplace_back(nets[id]);
    vector<vector<int>> batches = getBatches(routers,netIndices);

    std::unordered_map<int,PatternRoute> PatternRoutes;
    for (const int netIndex : netIndices) {
        PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
        patternRoute.constructSteinerTree();
        patternRoute.constructRoutingDAG();
        PatternRoutes.insert(std::make_pair(netIndex,patternRoute));
    }

    // int cnt = 0;
    std::mutex mtx;
    for (const vector<int>& batch : batches) {
        runJobsMT(batch.size(),numofThreads, [&](int jobIdx) {
            auto patternRoute = PatternRoutes.find(batch[jobIdx])->second;
            patternRoute.run();
            mtx.lock();
            gridGraph.commitTree(nets[batch[jobIdx]].getRoutingTree());
            mtx.unlock();
        });
        // std::cout<<"batch"<<cnt<<" has complete!"<<"\n";
        // cnt ++;
    }

    // for (const int netIndex : netIndices) {
    //     PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
    //     patternRoute.constructSteinerTree();
    //     patternRoute.constructRoutingDAG();
    //     patternRoute.run();
    //     gridGraph.commitTree(nets[netIndex].getRoutingTree());
    // }
    
    netIndices.clear();
    for (const auto& net : nets) {
        int netOverflow=gridGraph.checkOverflow(net.getRoutingTree());
        if (netOverflow > 0) {
            netIndices.push_back(net.getIndex());
            netOverflows[net.getIndex()]=netOverflow;
        }
    }
    log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
    logeol();
    
    t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    
    // Stage 2: Pattern routing with possible detours
    n2 = netIndices.size();
    if (netIndices.size() > 0) {
        log() << "stage 2: pattern routing with possible detours" << std::endl;
        GridGraphView<bool> congestionView; // (2d) direction -> x -> y -> has overflow?
        gridGraph.extractCongestionView(congestionView);
        // for (const int netIndex : netIndices) {
        //     GRNet& net = nets[netIndex];
        //     gridGraph.commitTree(net.getRoutingTree(), true);
        // }
        // sortNetIndices(netIndices);
        sortNetIndicesOFDALD(netIndices, netOverflows);
        // sortNetIndicesD(netIndices);
        // sortNetIndicesOLD(netIndices);
        // sortNetIndicesOLI(netIndices);
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            gridGraph.commitTree(net.getRoutingTree(), true);
            PatternRoute patternRoute(net, gridGraph, parameters);
            patternRoute.constructSteinerTree();
            patternRoute.constructRoutingDAG();
            patternRoute.constructDetours(congestionView); // KEY DIFFERENCE compared to stage 1
            patternRoute.run();
            gridGraph.commitTree(net.getRoutingTree());
        }
        
        netIndices.clear();
        // add net overflow>0 to netIndices to maze routing
        for (const auto& net : nets) {
            int netOverflow=gridGraph.checkOverflow(net.getRoutingTree());
            if ( netOverflow > 0) {
                netIndices.push_back(net.getIndex());
                netOverflows[net.getIndex()]=netOverflow;
                // log() << "netindex: " << net.getIndex() << " netoverflow: " << netOverflow << std::endl;
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        logeol();
    }
    
    t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    
    // Stage 3: maze routing on sparsified routing graph
    n3 = netIndices.size();
    if (netIndices.size() > 0) {
        log() << "stage 3: maze routing on sparsified routing graph" << std::endl;
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            gridGraph.commitTree(net.getRoutingTree(), true);
        }
        GridGraphView<CostT> wireCostView;
        gridGraph.extractWireCostView(wireCostView);
        // sortNetIndicesOFD(netIndices, netOverflows);
        sortNetIndicesOFDALD(netIndices, netOverflows);
        // sortNetIndices(netIndices);
        // sortNetIndicesOLD(netIndices);
        // sortNetIndicesOLI(netIndices);
        // sortNetIndicesD(netIndices);
        SparseGrid grid(10, 10, 0, 0);
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            // log() << "netindex: " << netIndex << " netoverflow:" << netOverflows[netIndex] << std::endl;
            // log() << "netindex: " << netIndex << " netoverlapScore:" << netScores[netIndex] << std::endl;
            // log() << "netindex: " << netIndex << " score:" << 30*netOverflows[netIndex]+net.getBoundingBox().hp() << std::endl;
            // gridGraph.commitTree(net.getRoutingTree(), true);
            // gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
            MazeRoute mazeRoute(net, gridGraph, parameters);
            mazeRoute.constructSparsifiedGraph(wireCostView, grid);
            mazeRoute.run();
            std::shared_ptr<SteinerTreeNode> tree = mazeRoute.getSteinerTree();
            assert(tree != nullptr);
            
            PatternRoute patternRoute(net, gridGraph, parameters);
            patternRoute.setSteinerTree(tree);
            patternRoute.constructRoutingDAG();
            patternRoute.run();
            
            gridGraph.commitTree(net.getRoutingTree());
            gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
            grid.step();
        }
        netIndices.clear();
        for (const auto& net : nets) {
            if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
                netIndices.push_back(net.getIndex());
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        logeol();
    }
    
    t3 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    
    // std::cout << "iteration statistics " 
    //     << n1 << " " << std::setprecision(3) << std::fixed << t1 << " " 
    //     << n2 << " " << std::setprecision(3) << std::fixed << t2 << " " 
    //     << n3 << " " << std::setprecision(3) << std::fixed << t3 << std::endl;

    log() << "phase1 runtime: " << std::setprecision(3) << std::fixed << t1 << std::endl;
    log() << "phase2 runtime: " << std::setprecision(3) << std::fixed << t2 << std::endl;
    log() << "phase3 runtime: " << std::setprecision(3) << std::fixed << t3 << std::endl;
    log() << "total runtime: " << std::setprecision(3) << std::fixed << t1+t2+t3 << std::endl;
    
    std::ofstream  afile;
    afile.open("time", std::ios::app);
    afile <<t1 <<" "<< t2 <<" "<< t3 <<" ";
    afile.close();

    printStatistics();
    if (parameters.write_heatmap) gridGraph.write();
}

void GlobalRouter::netSortRoute(int cycles) {

    for(int i=0; i<cycles ;i++){
        int n1 = 0, n2 = 0, n3 = 0;
        double t1 = 0, t2 = 0, t3 = 0;
        auto t = std::chrono::high_resolution_clock::now();
        
        vector<int> netIndices;
        vector<int> netOverflows(nets.size());
        netIndices.reserve(nets.size());
        for (const auto& net : nets) netIndices.push_back(net.getIndex());
        log() << "Route Index: " << i << std::endl;
        // Stage 1: Pattern routing
        n1 = netIndices.size();
        PatternRoute::readFluteLUT();
        // log() << "stage 1: pattern routing" << std::endl;
        sortNetIndicesRandom(netIndices);
        for (const int netIndex : netIndices) {
            PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
            patternRoute.constructSteinerTree();
            patternRoute.constructRoutingDAG();
            patternRoute.run();
            gridGraph.commitTree(nets[netIndex].getRoutingTree());
        }
        
        netIndices.clear();
        for (const auto& net : nets) {
            int netOverflow=gridGraph.checkOverflow(net.getRoutingTree());
            if (netOverflow > 0) {
                netIndices.push_back(net.getIndex());
                netOverflows[net.getIndex()]=netOverflow;
            }
        }
        // log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        // logeol();
        
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
        t = std::chrono::high_resolution_clock::now();
        
        // Stage 2: Pattern routing with possible detours
        n2 = netIndices.size();
        if (netIndices.size() > 0) {
            // log() << "stage 2: pattern routing with possible detours" << std::endl;
            GridGraphView<bool> congestionView; // (2d) direction -> x -> y -> has overflow?
            gridGraph.extractCongestionView(congestionView);
            // for (const int netIndex : netIndices) {
            //     GRNet& net = nets[netIndex];
            //     gridGraph.commitTree(net.getRoutingTree(), true);
            // }
            // sortNetIndices(netIndices);
            sortNetIndicesRandom(netIndices);
            for (const int netIndex : netIndices) {
                GRNet& net = nets[netIndex];
                gridGraph.commitTree(net.getRoutingTree(), true);
                PatternRoute patternRoute(net, gridGraph, parameters);
                patternRoute.constructSteinerTree();
                patternRoute.constructRoutingDAG();
                patternRoute.constructDetours(congestionView); // KEY DIFFERENCE compared to stage 1
                patternRoute.run();
                gridGraph.commitTree(net.getRoutingTree());
            }
            
            netIndices.clear();
            // add net overflow>0 to netIndices to maze routing
            for (const auto& net : nets) {
                int netOverflow=gridGraph.checkOverflow(net.getRoutingTree());
                if ( netOverflow > 0) {
                    netIndices.push_back(net.getIndex());
                    netOverflows[net.getIndex()]=netOverflow;
                    // log() << "netindex: " << net.getIndex() << " netoverflow: " << netOverflow << std::endl;
                }
            }
            // log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
            // logeol();
        }
        
        t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
        t = std::chrono::high_resolution_clock::now();
        
        // Stage 3: maze routing on sparsified routing graph
        n3 = netIndices.size();
        if (netIndices.size() > 0) {
            // log() << "stage 3: maze routing on sparsified routing graph" << std::endl;
            for (const int netIndex : netIndices) {
                GRNet& net = nets[netIndex];
                gridGraph.commitTree(net.getRoutingTree(), true);
            }
            GridGraphView<CostT> wireCostView;
            gridGraph.extractWireCostView(wireCostView);
            sortNetIndicesRandom(netIndices);
            SparseGrid grid(10, 10, 0, 0);
            for (const int netIndex : netIndices) {
                GRNet& net = nets[netIndex];
                // gridGraph.commitTree(net.getRoutingTree(), true);
                // gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
                MazeRoute mazeRoute(net, gridGraph, parameters);
                mazeRoute.constructSparsifiedGraph(wireCostView, grid);
                mazeRoute.run();
                std::shared_ptr<SteinerTreeNode> tree = mazeRoute.getSteinerTree();
                assert(tree != nullptr);
                
                PatternRoute patternRoute(net, gridGraph, parameters);
                patternRoute.setSteinerTree(tree);
                patternRoute.constructRoutingDAG();
                patternRoute.run();
                
                gridGraph.commitTree(net.getRoutingTree());
                gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
                grid.step();
            }
            netIndices.clear();
            for (const auto& net : nets) {
                if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
                    netIndices.push_back(net.getIndex());
                }
            }
            // log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
            // logeol();
        }
        
        t3 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
        t = std::chrono::high_resolution_clock::now();
        
        // std::cout << "iteration statistics " 
        //     << n1 << " " << std::setprecision(3) << std::fixed << t1 << " " 
        //     << n2 << " " << std::setprecision(3) << std::fixed << t2 << " " 
        //     << n3 << " " << std::setprecision(3) << std::fixed << t3 << std::endl;

        // log() << "phase1 runtime: " << std::setprecision(3) << std::fixed << t1 << std::endl;
        // log() << "phase2 runtime: " << std::setprecision(3) << std::fixed << t2 << std::endl;
        // log() << "phase3 runtime: " << std::setprecision(3) << std::fixed << t3 << std::endl;
        // log() << "total runtime: " << std::setprecision(3) << std::fixed << t1+t2+t3 << std::endl;
        
        printStatistics();
        if (parameters.write_heatmap) gridGraph.write();

    }

}

void GlobalRouter::sortNetIndices(vector<int>& netIndices) const {
    vector<int> halfParameters(nets.size());
    for (int netIndex : netIndices) {
        auto& net = nets[netIndex];
        halfParameters[netIndex] = net.getBoundingBox().hp();
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return halfParameters[lhs] < halfParameters[rhs];
    });
}

void GlobalRouter::sortNetIndicesD(vector<int>& netIndices) const {
    vector<int> halfParameters(nets.size());
    for (int netIndex : netIndices) {
        auto& net = nets[netIndex];
        halfParameters[netIndex] = net.getBoundingBox().hp();
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return halfParameters[lhs] > halfParameters[rhs];
    });
}

void GlobalRouter::sortNetIndicesOFD(vector<int>& netIndices, vector<int>& netOverflow) const {
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return netOverflow[lhs] > netOverflow[rhs];
    });
}

void GlobalRouter::sortNetIndicesOFDALD(vector<int>& netIndices, vector<int>& netOverflow) const {
    vector<int> scores(nets.size());
    for (int netIndex : netIndices) {
        auto& net = nets[netIndex];
        scores[netIndex] = net.getBoundingBox().hp() + 30*netOverflow[netIndex];
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });
}

void GlobalRouter::sortNetIndicesOLD(vector<int>& netIndices) const {
    vector<int> scores(nets.size());
    for (int netIndex : netIndices) {
        int overlapNum=0;
        auto& net = nets[netIndex];
        for (int otherNetIndex : netIndices) {
            if (otherNetIndex == netIndex) continue;
            auto& otherNet = nets[otherNetIndex];
            bool overlap = net.overlap(otherNet);
            if (overlap) overlapNum++;
        }
        int halfParameter = net.getBoundingBox().hp();
        scores[netIndex] = overlapNum/(halfParameter+2);
        // netScores[netIndex] = overlapNum/halfParameter+2;
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

}

void GlobalRouter::sortNetIndicesOLI(vector<int>& netIndices) const {
    vector<int> scores(nets.size());
    for (int netIndex : netIndices) {
        int overlapNum=0;
        auto& net = nets[netIndex];
        for (int otherNetIndex : netIndices) {
            if (otherNetIndex == netIndex) continue;
            auto& otherNet = nets[otherNetIndex];
            bool overlap = net.overlap(otherNet);
            if (overlap) overlapNum++;
        }
        int halfParameter = net.getBoundingBox().hp();
        scores[netIndex] = overlapNum/(halfParameter+2);
        // netScores[netIndex] = overlapNum/halfParameter+2;
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

}

void GlobalRouter::sortNetIndicesRandom(vector<int>& netIndices) const {
    // 使用 std::random_device 获取随机种子
    std::random_device rd;
    
    // 使用随机种子初始化 std::mt19937 引擎
    std::mt19937 gen(rd());

    // 使用 std::shuffle 对 vector 元素进行随机排序
    std::shuffle(netIndices.begin(), netIndices.end(), gen);
}

void GlobalRouter::getGuides(const GRNet& net, vector<std::pair<int, utils::BoxT<int>>>& guides) {
    auto& routingTree = net.getRoutingTree();
    if (!routingTree) return;
    // 0. Basic guides
    GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
        for (const auto& child : node->children) {
            if (node->layerIdx == child->layerIdx) {
                guides.emplace_back(
                    node->layerIdx, utils::BoxT<int>(
                        min(node->x, child->x), min(node->y, child->y),
                        max(node->x, child->x), max(node->y, child->y)
                    )
                );
            } else {
                int maxLayerIndex = max(node->layerIdx, child->layerIdx);
                for (int layerIdx = min(node->layerIdx, child->layerIdx); layerIdx <= maxLayerIndex; layerIdx++) {
                    guides.emplace_back(layerIdx, utils::BoxT<int>(node->x, node->y));
                }
            }
        }
    });
    
    
    auto getSpareResource = [&] (const GRPoint& point) {
        double resource = std::numeric_limits<double>::max();
        unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
        if (point[direction] + 1 < gridGraph.getSize(direction)) {
            resource = min(resource, gridGraph.getEdge(point.layerIdx, point.x, point.y).getResource());
        }
        if (point[direction] > 0) {
            GRPoint lower = point;
            lower[direction] -= 1;
            resource = min(resource, gridGraph.getEdge(lower.layerIdx, point.x, point.y).getResource());
        }
        return resource;
    };
    
    // 1. Pin access patches
    assert(parameters.min_routing_layer + 1 < gridGraph.getNumLayers());
    for (auto& gpts : net.getPinAccessPoints()) {
        for (auto& gpt : gpts) {
            if (gpt.layerIdx < parameters.min_routing_layer) {
                int padding = 0;
                if (getSpareResource({parameters.min_routing_layer, gpt.x, gpt.y}) < parameters.pin_patch_threshold) {
                    padding = parameters.pin_patch_padding;
                }
                for (int layerIdx = gpt.layerIdx; layerIdx <= parameters.min_routing_layer + 1; layerIdx++) {
                    guides.emplace_back(layerIdx, utils::BoxT<int>(
                        max(gpt.x - padding, 0),
                        max(gpt.y - padding, 0),
                        min(gpt.x + padding, (int)gridGraph.getSize(0) - 1),
                        min(gpt.y + padding, (int)gridGraph.getSize(1) - 1)
                    ));
                    areaOfPinPatches += (guides.back().second.x.range() + 1) * (guides.back().second.y.range() + 1);
                }
            }
        }
    }
    
    // 2. Wire segment patches
    GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
        for (const auto& child : node->children) {
            if (node->layerIdx == child->layerIdx) {
                double wire_patch_threshold = parameters.wire_patch_threshold;
                unsigned direction = gridGraph.getLayerDirection(node->layerIdx);
                int l = min((*node)[direction], (*child)[direction]);
                int h = max((*node)[direction], (*child)[direction]);
                int r = (*node)[1 - direction];
                for (int c = l; c <= h; c++) {
                    bool patched = false;
                    GRPoint point = (direction == MetalLayer::H ? GRPoint(node->layerIdx, c, r) : GRPoint(node->layerIdx, r, c));
                    if (getSpareResource(point) < wire_patch_threshold) {
                        for (int layerIndex = node->layerIdx - 1; layerIndex <= node->layerIdx + 1; layerIndex += 2) {
                            if (layerIndex < parameters.min_routing_layer || layerIndex >= gridGraph.getNumLayers()) continue;
                            if (getSpareResource({layerIndex, point.x, point.y}) >= 1.0) {
                                guides.emplace_back(layerIndex, utils::BoxT<int>(point.x, point.y));
                                areaOfWirePatches += 1;
                                patched = true;
                            }
                        }
                    } 
                    if (patched) {
                        wire_patch_threshold = parameters.wire_patch_threshold;
                    } else {
                        wire_patch_threshold *= parameters.wire_patch_inflation_rate;
                    }
                }
            }
        }
    });
}

void GlobalRouter::printStatistics() const {
    log() << "routing statistics" << std::endl;
    loghline();

    // wire length and via count
    uint64_t wireLength = 0;
    int viaCount = 0;
    vector<vector<vector<int>>> wireUsage;
    wireUsage.assign(
        gridGraph.getNumLayers(), vector<vector<int>>(gridGraph.getSize(0), vector<int>(gridGraph.getSize(1), 0))
    );
    for (const auto& net : nets) {
        GRTreeNode::preorder(net.getRoutingTree(), [&] (std::shared_ptr<GRTreeNode> node) {
            for (const auto& child : node->children) {
                if (node->layerIdx == child->layerIdx) {
                    unsigned direction = gridGraph.getLayerDirection(node->layerIdx);
                    int l = min((*node)[direction], (*child)[direction]);
                    int h = max((*node)[direction], (*child)[direction]);
                    int r = (*node)[1 - direction];
                    for (int c = l; c < h; c++) {
                        wireLength += gridGraph.getEdgeLength(direction, c);
                        int x = direction == MetalLayer::H ? c : r;
                        int y = direction == MetalLayer::H ? r : c;
                        wireUsage[node->layerIdx][x][y] += 1;
                    }
                } else {
                    viaCount += abs(node->layerIdx - child->layerIdx);
                }
            }
        });
    }
    
    // resource
    CapacityT overflow = 0;

    CapacityT minResource = std::numeric_limits<CapacityT>::max();
    GRPoint bottleneck(-1, -1, -1);
    for (int layerIndex = parameters.min_routing_layer; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
        unsigned direction = gridGraph.getLayerDirection(layerIndex);
        for (int x = 0; x < gridGraph.getSize(0) - 1 + direction; x++) {
            for (int y = 0; y < gridGraph.getSize(1) - direction; y++) {
                CapacityT resource = gridGraph.getEdge(layerIndex, x, y).getResource();
                if (resource < minResource) {
                    minResource = resource;
                    bottleneck = {layerIndex, x, y};
                }
                CapacityT usage = wireUsage[layerIndex][x][y];
                CapacityT capacity = max(gridGraph.getEdge(layerIndex, x, y).capacity, 0.0);
                if (usage > 0.0 && usage > capacity) {
                    overflow += usage - capacity;
                }
            }
        }
    }
    
    // log() << "wire length (metric):  " << wireLength / gridGraph.getM2Pitch() << std::endl;
    log() << "wire length (metric):  " << wireLength << std::endl;
    log() << "total via count:       " << viaCount << std::endl;
    log() << "total wire overflow:   " << (int)overflow << std::endl;
    logeol();

    log() << "min resource: " << minResource << std::endl;
    log() << "bottleneck:   " << bottleneck << std::endl;

    logeol();
}

void GlobalRouter::write(std::string guide_file) {
    log() << "generating route guides..." << std::endl;
    if (guide_file == "") guide_file = parameters.out_file;
    
    areaOfPinPatches = 0;
    areaOfWirePatches = 0;
    std::stringstream ss;
    for (const GRNet& net : nets) {
        vector<std::pair<int, utils::BoxT<int>>> guides;
        getGuides(net, guides);
        
        ss << net.getName() << std::endl;
        ss << "(" << std::endl;
        // write guides in accurate coordinates
        // for (const auto& guide : guides) {
        //     ss << gridGraph.getGridline(0, guide.second.x.low) << " "
        //          << gridGraph.getGridline(1, guide.second.y.low) << " "
        //          << gridGraph.getGridline(0, guide.second.x.high + 1) << " "
        //          << gridGraph.getGridline(1, guide.second.y.high + 1) << " "
        //          << gridGraph.getLayerName(guide.first) << std::endl;
        // }
        // write guides in grid coordinates
        for (const auto& guide : guides) {
            ss << guide.second.x.low << " "
                 << guide.second.y.low << " "
                 << guide.second.x.high + 1 << " "
                 << guide.second.y.high + 1 << " "
                 << gridGraph.getLayerName(guide.first) << std::endl;
        }
        ss << ")" << std::endl;
    }
    log() << "total area of pin access patches: " << areaOfPinPatches << std::endl;
    log() << "total area of wire segment patches: " << areaOfWirePatches << std::endl;
    log() << std::endl;
    log() << "writing output..." << std::endl;
    std::ofstream fout(guide_file);
    fout << ss.str();
    fout.close();
    log() << "finished writing output..." << std::endl;
}

vector<vector<int>> GlobalRouter::getBatches(vector<SingleNetRouter>& routers, const vector<int>& netsToRoute) {
    vector<int> batch(netsToRoute.size());
    if (numofThreads == 1)
    {
        vector<vector<int>> batches;
        batches.emplace_back(netsToRoute);
        return batches;
    }
    
    for (int i = 0; i < netsToRoute.size(); i++) batch[i] = i;

    runJobsMT(batch.size(), numofThreads, [&](int jobIdx) {
        auto& router = routers[batch[jobIdx]];
        const auto mergedPinAccessBoxes = nets[netsToRoute[jobIdx]].getPinAccessPoints();
        utils::IntervalT<long int> xIntvl, yIntvl;
        for (auto& points : mergedPinAccessBoxes) {
            for (auto& point : points) {
                xIntvl.Update(point[X]);
                yIntvl.Update(point[Y]);
            }
        }
        router.guides.emplace_back(0, xIntvl, yIntvl);
    });
    Scheduler scheduler(routers,gridGraph.getNumLayers());
    const vector<vector<int>>& batches = scheduler.scheduleOrderEq(numofThreads);

    return batches;
}

void GlobalRouter::runJobsMT(int numJobs, int numofThreads, const std::function<void(int)>& handle) {
    int numThreads = min(numJobs, numofThreads);
    if (numThreads <= 1) {
        for (int i = 0; i < numJobs; ++i) {
            handle(i);
        }
    } else {
        int globalJobIdx = 0;
        std::mutex mtx;
        auto thread_func = [&](int threadIdx) {
            int jobIdx;
            while (true) {
                mtx.lock();
                jobIdx = globalJobIdx++;
                mtx.unlock();
                if (jobIdx >= numJobs) {
                    break;
                }
                handle(jobIdx);
            }
        };

        std::thread threads[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = std::thread(thread_func, i);
        }
        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }
    }
}