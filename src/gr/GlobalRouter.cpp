#include "GlobalRouter.h"
#include "PatternRoute.h"
#include "MazeRoute.h"
#include "GPUMazeRoute.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include "GridGraph.h"
#include "../obj/ISPD24Parser.h"
#include <cmath>
#include <fstream>

using std::max;
using std::min;
using std::vector;

GlobalRouter::GlobalRouter(const ISPD24Parser &parser, const Parameters &params)
    : gridGraph(parser, params), parameters(params)
{
    nets.reserve(parser.net_names.size());
    for (int i = 0; i < parser.net_names.size(); i++)
        nets.emplace_back(i, parser.net_names[i], parser.net_access_points[i]);
    unit_length_wire_cost = parser.unit_length_wire_cost;
    unit_via_cost = parser.unit_via_cost;
    unit_length_short_costs = parser.unit_length_short_costs;
}

void GlobalRouter::route()
{
    int n1 = 0, n2 = 0, n3 = 0;
    double t1 = 0, t2 = 0, t3 = 0;

    auto t = std::chrono::high_resolution_clock::now();

    vector<int> netIndices;
    vector<int> netOverflows(nets.size());
    netIndices.reserve(nets.size());
    for (const auto &net : nets)
        netIndices.push_back(net.getIndex());
    // Stage 1: Pattern routing
    n1 = netIndices.size();
    PatternRoute::readFluteLUT();
    log() << "stage 1: pattern routing" << std::endl;
    sortNetIndices(netIndices);
    for (const int netIndex : netIndices)
    {
        PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
        patternRoute.constructSteinerTree();
        patternRoute.constructRoutingDAG();
        patternRoute.run();
        gridGraph.commitTree(nets[netIndex].getRoutingTree());
    }

    netIndices.clear();
    for (const auto &net : nets)
    {
        int netOverflow = gridGraph.checkOverflow(net.getRoutingTree());
        if (netOverflow > 0)
        {
            netIndices.push_back(net.getIndex());
            netOverflows[net.getIndex()] = netOverflow;
        }
    }
    log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
    logeol();
    // printStatistics();
    t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();

    // Stage 2: Pattern routing with possible detours
    n2 = netIndices.size();
    if (netIndices.size() > 0)
    {
        log() << "stage 2: pattern routing with possible detours" << std::endl;
        GridGraphView<bool> congestionView; // (2d) direction -> x -> y -> has overflow?
        gridGraph.extractCongestionView(congestionView);
        // for (const int netIndex : netIndices) {
        //     GRNet& net = nets[netIndex];
        //     gridGraph.commitTree(net.getRoutingTree(), true);
        // }
#ifndef ENABLE_ISSSORT
        sortNetIndices(netIndices);
#else
        log() << "sort net indices with OFDALD" << std::endl;
        sortNetIndicesOFDALD(netIndices, netOverflows);
        // sortNetIndicesD(netIndices);
#endif
        for (const int netIndex : netIndices)
        {
            GRNet &net = nets[netIndex];
            gridGraph.commitTree(net.getRoutingTree(), true);
            PatternRoute patternRoute(net, gridGraph, parameters);
            patternRoute.constructSteinerTree();
            patternRoute.constructRoutingDAG();
            patternRoute.constructDetours(congestionView); // KEY DIFFERENCE compared to stage 1
            patternRoute.run();
            gridGraph.commitTree(net.getRoutingTree());
        }

        netIndices.clear();
        for (const auto &net : nets)
        {
            int netOverflow = gridGraph.checkOverflow(net.getRoutingTree());
            if (netOverflow > 0)
            {
                netIndices.push_back(net.getIndex());
                netOverflows[net.getIndex()] = netOverflow;
                // log() << "netindex: " << net.getIndex() << " netoverflow: " << netOverflow << std::endl;
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        logeol();
    }
    // printStatistics();
    t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();

    // Stage 3: maze routing
    n3 = netIndices.size();
#ifndef ENABLE_CUDA
    if (netIndices.size() > 0)
    {
        log() << "stage 3: maze routing on sparsified routing graph" << std::endl;
        for (const int netIndex : netIndices)
        {
            GRNet &net = nets[netIndex];
            gridGraph.commitTree(net.getRoutingTree(), true);
        }
        GridGraphView<CostT> wireCostView;
        gridGraph.extractWireCostView(wireCostView);
#ifndef ENABLE_ISSSORT
        sortNetIndices(netIndices);
#else
        log() << "sort net indices with OFDALD" << std::endl;
        sortNetIndicesOFDALD(netIndices, netOverflows);
#endif
        SparseGrid grid(10, 10, 0, 0);
        for (const int netIndex : netIndices)
        {
            GRNet &net = nets[netIndex];
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
        for (const auto &net : nets)
        {
            if (gridGraph.checkOverflow(net.getRoutingTree()) > 0)
            {
                netIndices.push_back(net.getIndex());
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        logeol();
    }
#else
    if (netIndices.size() > 0)
    {
        std::vector<bool> isGamerRoutedNet(nets.size(), false);
        log() << "stage 3: gpu maze routing\n";
        GPUMazeRoute gamer(nets, gridGraph, parameters);
        log() << "gamer init. overflow net: " << netIndices.size() << "/" << nets.size() << "\n";
        for (int iter = 1; iter <= 3 && netIndices.size() > 0; iter++)
        {
#ifndef ENABLE_ISSSORT
            sortNetIndices(netIndices);
#else
            log() << "sort net indices with OFDALD" << std::endl;
            sortNetIndicesOFDALD(netIndices, netOverflows);
#endif
            gamer.route(netIndices, 3 + iter, 10 * iter);
            for (auto netId : netIndices)
                isGamerRoutedNet[netId] = true;
            gamer.getOverflowNetIndices(netIndices);
            log() << "gamer iter " << iter << " overflow net: " << netIndices.size() << "/" << nets.size() << "\n";
        }
        netIndices.clear();
        for (int netId = 0; netId < nets.size(); netId++)
            if (isGamerRoutedNet[netId])
                netIndices.push_back(netId);
        log() << "commiting gamer's result ...\n";
        gamer.apply(netIndices);
        log() << "commiting done\n";
    }
#endif
    t3 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();

    log() << "step routed #nets: " << n1 << ", " << n2 << ", " << n3 << "\n";
    log() << "step time consumption: "
          << std::setprecision(3) << std::fixed << t1 << " s, "
          << std::setprecision(3) << std::fixed << t2 << " s, "
          << std::setprecision(3) << std::fixed << t3 << " s\n";

    printStatistics();
    if (parameters.write_heatmap)
        gridGraph.write();
}

void GlobalRouter::sortNetIndices(vector<int> &netIndices) const
{
    vector<int> halfParameters(nets.size());
    for (int netIndex : netIndices)
    {
        auto &net = nets[netIndex];
        halfParameters[netIndex] = net.getBoundingBox().hp();
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return halfParameters[lhs] < halfParameters[rhs]; });
}

void GlobalRouter::sortNetIndicesD(vector<int> &netIndices) const
{
    vector<int> halfParameters(nets.size());
    for (int netIndex : netIndices)
    {
        auto &net = nets[netIndex];
        halfParameters[netIndex] = net.getBoundingBox().hp();
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return halfParameters[lhs] > halfParameters[rhs]; });
}

void GlobalRouter::sortNetIndicesOFD(vector<int> &netIndices, vector<int> &netOverflow) const
{
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return netOverflow[lhs] > netOverflow[rhs]; });
}

void GlobalRouter::sortNetIndicesOFDALD(vector<int> &netIndices, vector<int> &netOverflow) const
{
    vector<int> scores(nets.size());
    for (int netIndex : netIndices)
    {
        auto &net = nets[netIndex];
        scores[netIndex] = net.getBoundingBox().hp() + 30 * netOverflow[netIndex];
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return scores[lhs] > scores[rhs]; });
}

void GlobalRouter::sortNetIndicesOLD(vector<int> &netIndices) const
{
    vector<int> scores(nets.size());
    for (int netIndex : netIndices)
    {
        int overlapNum = 0;
        auto &net = nets[netIndex];
        for (int otherNetIndex : netIndices)
        {
            if (otherNetIndex == netIndex)
                continue;
            auto &otherNet = nets[otherNetIndex];
            bool overlap = net.overlap(otherNet);
            if (overlap)
                overlapNum++;
        }
        int halfParameter = net.getBoundingBox().hp();
        scores[netIndex] = overlapNum / (halfParameter + 2);
        // netScores[netIndex] = overlapNum/halfParameter+2;
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return scores[lhs] > scores[rhs]; });
}

void GlobalRouter::sortNetIndicesOLI(vector<int> &netIndices) const
{
    vector<int> scores(nets.size());
    for (int netIndex : netIndices)
    {
        int overlapNum = 0;
        auto &net = nets[netIndex];
        for (int otherNetIndex : netIndices)
        {
            if (otherNetIndex == netIndex)
                continue;
            auto &otherNet = nets[otherNetIndex];
            bool overlap = net.overlap(otherNet);
            if (overlap)
                overlapNum++;
        }
        int halfParameter = net.getBoundingBox().hp();
        scores[netIndex] = overlapNum / (halfParameter + 2);
        // netScores[netIndex] = overlapNum/halfParameter+2;
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs)
         { return scores[lhs] > scores[rhs]; });
}

// void GlobalRouter::sortNetIndicesRandom(vector<int>& netIndices) const {
//     // 使用 std::random_device 获取随机种子
//     std::random_device rd;

//     // 使用随机种子初始化 std::mt19937 引擎
//     std::mt19937 gen(rd());

//     // 使用 std::shuffle 对 vector 元素进行随机排序
//     std::shuffle(netIndices.begin(), netIndices.end(), gen);
// }

// void GlobalRouter::getGuides(const GRNet& net, vector<std::pair<int, utils::BoxT<int>>>& guides) {
//     auto& routingTree = net.getRoutingTree();
//     if (!routingTree) return;
//     // 0. Basic guides
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

//     auto getSpareResource = [&] (const GRPoint& point) {
//         double resource = std::numeric_limits<double>::max();
//         unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
//         if (point[direction] + 1 < gridGraph.getSize(direction)) {
//             resource = min(resource, gridGraph.getEdge(point.layerIdx, point.x, point.y).getResource());
//         }
//         if (point[direction] > 0) {
//             GRPoint lower = point;
//             lower[direction] -= 1;
//             resource = min(resource, gridGraph.getEdge(lower.layerIdx, point.x, point.y).getResource());
//         }
//         return resource;
//     };

//     // 1. Pin access patches
//     assert(parameters.min_routing_layer + 1 < gridGraph.getNumLayers());
//     for (auto& gpts : net.getPinAccessPoints()) {
//         for (auto& gpt : gpts) {
//             if (gpt.layerIdx < parameters.min_routing_layer) {
//                 int padding = 0;
//                 if (getSpareResource({parameters.min_routing_layer, gpt.x, gpt.y}) < parameters.pin_patch_threshold) {
//                     padding = parameters.pin_patch_padding;
//                 }
//                 for (int layerIdx = gpt.layerIdx; layerIdx <= parameters.min_routing_layer + 1; layerIdx++) {
//                     guides.emplace_back(layerIdx, utils::BoxT<int>(
//                         max(gpt.x - padding, 0),
//                         max(gpt.y - padding, 0),
//                         min(gpt.x + padding, (int)gridGraph.getSize(0) - 1),
//                         min(gpt.y + padding, (int)gridGraph.getSize(1) - 1)
//                     ));
//                     areaOfPinPatches += (guides.back().second.x.range() + 1) * (guides.back().second.y.range() + 1);
//                 }
//             }
//         }
//     }

//     // 2. Wire segment patches
//     GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
//         for (const auto& child : node->children) {
//             if (node->layerIdx == child->layerIdx) {
//                 double wire_patch_threshold = parameters.wire_patch_threshold;
//                 unsigned direction = gridGraph.getLayerDirection(node->layerIdx);
//                 int l = min((*node)[direction], (*child)[direction]);
//                 int h = max((*node)[direction], (*child)[direction]);
//                 int r = (*node)[1 - direction];
//                 for (int c = l; c <= h; c++) {
//                     bool patched = false;
//                     GRPoint point = (direction == MetalLayer::H ? GRPoint(node->layerIdx, c, r) : GRPoint(node->layerIdx, r, c));
//                     if (getSpareResource(point) < wire_patch_threshold) {
//                         for (int layerIndex = node->layerIdx - 1; layerIndex <= node->layerIdx + 1; layerIndex += 2) {
//                             if (layerIndex < parameters.min_routing_layer || layerIndex >= gridGraph.getNumLayers()) continue;
//                             if (getSpareResource({layerIndex, point.x, point.y}) >= 1.0) {
//                                 guides.emplace_back(layerIndex, utils::BoxT<int>(point.x, point.y));
//                                 areaOfWirePatches += 1;
//                                 patched = true;
//                             }
//                         }
//                     }
//                     if (patched) {
//                         wire_patch_threshold = parameters.wire_patch_threshold;
//                     } else {
//                         wire_patch_threshold *= parameters.wire_patch_inflation_rate;
//                     }
//                 }
//             }
//         }
//     });
// }

void GlobalRouter::getGuides(const GRNet &net, vector<std::pair<std::pair<int, int>, utils::BoxT<int>>> &guides)
{
    auto &routingTree = net.getRoutingTree();
    if (!routingTree)
        return;
    // 0. Basic guides
    GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node)
                         {
        for (const auto& child : node->children) {
            if (node->layerIdx == child->layerIdx) {
                if(min(node->x, child->x)!=max(node->x, child->x) || min(node->y, child->y)!=max(node->y, child->y)){
                    guides.emplace_back(
                        std::make_pair(node->layerIdx,node->layerIdx), utils::BoxT<int>(
                            min(node->x, child->x), min(node->y, child->y),
                            max(node->x, child->x), max(node->y, child->y)
                        )
                    );
                }
            } else {
                int maxLayerIndex = max(node->layerIdx, child->layerIdx);
                int minLayerIndex = min(node->layerIdx, child->layerIdx);
                guides.emplace_back(std::make_pair(minLayerIndex, maxLayerIndex), utils::BoxT<int>(node->x, node->y));
            }
        } });

    auto getSpareResource = [&](const GRPoint &point)
    {
        double resource = std::numeric_limits<double>::max();
        unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
        if (point[direction] + 1 < gridGraph.getSize(direction))
        {
            resource = min(resource, gridGraph.getEdge(point.layerIdx, point.x, point.y).getResource());
        }
        if (point[direction] > 0)
        {
            GRPoint lower = point;
            lower[direction] -= 1;
            resource = min(resource, gridGraph.getEdge(lower.layerIdx, point.x, point.y).getResource());
        }
        return resource;
    };

}

void GlobalRouter::printStatistics() const
{
    log() << "routing statistics" << std::endl;
    loghline();

    // wire length and via count
    uint64_t wireLength = 0;
    int viaCount = 0;
    vector<vector<vector<int>>> wireUsage;
    vector<vector<vector<int>>> nonstack_via_counter;
    vector<vector<vector<int>>> flag;
    wireUsage.assign(
        gridGraph.getNumLayers(), vector<vector<int>>(gridGraph.getSize(0), vector<int>(gridGraph.getSize(1), 0)));
    nonstack_via_counter.assign(
        gridGraph.getNumLayers(), vector<vector<int>>(gridGraph.getSize(0), vector<int>(gridGraph.getSize(1), 0)));
    flag.assign(
        gridGraph.getNumLayers(), vector<vector<int>>(gridGraph.getSize(0), vector<int>(gridGraph.getSize(1), -1)));
    for (const auto &net : nets)
    {
        vector<vector<int>> via_loc;
        if (net.getRoutingTree() == nullptr)
        {
            log() << "ERROR: null GRTree net(id=" << net.getIndex() << "\n";
            exit(-1);
        }
        GRTreeNode::preorder(net.getRoutingTree(), [&](std::shared_ptr<GRTreeNode> node)
        {
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
                        flag[node->layerIdx][x][y] = net.getIndex();
                    }
                    int x = direction == MetalLayer::H ? h : r;
                    int y = direction == MetalLayer::H ? r : h;
                    flag[node->layerIdx][x][y] = net.getIndex();
                } else {
                        int minLayerIndex = min(node->layerIdx, child->layerIdx);
                        int maxLayerIndex = max(node->layerIdx, child->layerIdx);
                        for (int layerIdx = minLayerIndex; layerIdx < maxLayerIndex; layerIdx++) {
                            via_loc.push_back({node->x, node->y, layerIdx});
                        }
                    viaCount += abs(node->layerIdx - child->layerIdx);
                }
            } });
        update_nonstack_via_counter(net.getIndex(), via_loc, flag, nonstack_via_counter);
    }

    // resource
    CapacityT overflow_cost = 0;
    double overflow_slope = 0.5;

    CapacityT minResource = std::numeric_limits<CapacityT>::max();
    GRPoint bottleneck(-1, -1, -1);

    for (unsigned z = parameters.min_routing_layer; z < gridGraph.getNumLayers(); z++)
    {
        unsigned long long num_overflows = 0;
        unsigned long long total_wl = 0;
        double layer_overflows = 0;
        double overflow = 0;
        unsigned layer_nonstack_via_counter = 0;
        for (unsigned x = 0; x < gridGraph.getSize(0); x++)
        {
            for (unsigned y = 0; y < gridGraph.getSize(1); y++)
            {
                layer_nonstack_via_counter += nonstack_via_counter[z][x][y];
                int usage = 2 * wireUsage[z][x][y] + nonstack_via_counter[z][x][y];
                double capacity = max(gridGraph.getEdge(z, x, y).capacity, 0.0);
                if (usage > 2 * capacity)
                {
                    num_overflows += usage - 2 * capacity;
                }

                if (capacity > 0)
                {
                    overflow = double(usage) - 2 * double(capacity);
                    layer_overflows += exp((overflow / 2) * overflow_slope);
                }
                else if (capacity == 0 && usage > 0)
                {
                    layer_overflows += exp(1.5 * double(usage) * overflow_slope);
                }
                else if (capacity < 0)
                {
                    printf("Capacity error (%d, %d, %d)\n", x, y, z);
                }
            }
        }
        overflow_cost += layer_overflows * 0.1; // gg.unit_overflow_cost();
        // log() << "Layer = " << z << " layer_nonstack_via_counter: "<<layer_nonstack_via_counter<< ", num_overflows = " << num_overflows << ", layer_overflows = " << layer_overflows << ", overflow cost = " << overflow_cost << std::endl;
        log() << "Layer = " << z << ", num_overflows = " << num_overflows << ", layer_overflows = " << layer_overflows << ", overflow cost = " << overflow_cost << std::endl;
    }

    double via_cost_scale = 1.0;
    double overflow_cost_scale = 1.0;
    double wireCost = wireLength * unit_length_wire_cost;
    double viaCost = viaCount * unit_via_cost * via_cost_scale;
    double overflowCost = overflow_cost * overflow_cost_scale;
    double totalCost = wireCost + viaCost + overflowCost;

    log() << "wire cost:                " << wireCost << std::endl;
    log() << "via cost:                 " << viaCost << std::endl;
    log() << "overflow cost:            " << overflowCost << std::endl;
    log() << "total cost(ispd24 score): " << totalCost << std::endl;

    logeol();
}

void GlobalRouter::write(std::string guide_file)
{
    log() << "generating route guides..." << std::endl;
    if (guide_file == "")
        guide_file = parameters.out_file;

    areaOfPinPatches = 0;
    areaOfWirePatches = 0;
    std::stringstream ss;
    for (const GRNet &net : nets)
    {
        vector<std::pair<std::pair<int, int>, utils::BoxT<int>>> guides;
        getGuides(net, guides);

        ss << net.getName() << std::endl;
        ss << "(" << std::endl;
        for (const auto &guide : guides)
        {
            ss << guide.second.x.low << " "
               << guide.second.y.low << " "
               << guide.first.first << " "
               << guide.second.x.high << " "
               << guide.second.y.high << " "
               << guide.first.second << std::endl;
        }
        ss << ")" << std::endl;
    }
    log() << std::endl;
    log() << "writing output..." << std::endl;
    std::ofstream fout(guide_file);
    fout << ss.str();
    fout.close();
    log() << "finished writing output..." << std::endl;
}

void GlobalRouter::update_nonstack_via_counter(unsigned net_idx,
  const std::vector<vector<int>> &via_loc,
  std::vector< std::vector< std::vector<int> > > &flag,
  std::vector< std::vector< std::vector<int> > > &nonstack_via_counter) const
{
  for(const auto &pp : via_loc) {
    if(flag[pp[2]][pp[0]][pp[1]] != net_idx) {
      flag[pp[2]][pp[0]][pp[1]] = net_idx;

      int direction = gridGraph.getLayerDirection(pp[2]);
      int size = gridGraph.getSize(direction);
      if(direction == 0) {
        if ((pp[0] > 0) && (pp[0] < size - 1)) {
          nonstack_via_counter[pp[2]][pp[0]-1][pp[1]]++;
          nonstack_via_counter[pp[2]][pp[0]][pp[1]]++;
        } else if (pp[0] > 0 ) {
          nonstack_via_counter[pp[2]][pp[0]-1][pp[1]] += 2;
        } else if (pp[0] < size - 1) {
          nonstack_via_counter[pp[2]][pp[0]][pp[1]] += 2;
        }
      } else if (direction == 1) {
        if ((pp[1] > 0) && (pp[1] < size - 1)) {
          nonstack_via_counter[pp[2]][pp[0]][pp[1]-1]++;
          nonstack_via_counter[pp[2]][pp[0]][pp[1]]++;
        } else if (pp[1] > 0 ) {
          nonstack_via_counter[pp[2]][pp[0]][pp[1]-1] += 2;
        } else if (pp[1] < size - 1) {
          nonstack_via_counter[pp[2]][pp[0]][pp[1]] += 2;
        }
      }
    }

  }
}
