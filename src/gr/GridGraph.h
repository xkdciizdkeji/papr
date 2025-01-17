#pragma once
#include "global.h"
#include "obj/Design.h"
#include "GRTree.h"

using CapacityT = double;
class GRNet;
template <typename Type> class GridGraphView;

struct GraphEdge {
    CapacityT capacity;
    CapacityT demand;
    GraphEdge(): capacity(0), demand(0) {}
    CapacityT getResource() const { return capacity - demand; }
};

class GridGraph {
public:
    GridGraph(const Design& design, const Parameters& params);
    inline DBU getLibDBU() const { return libDBU; }
    inline DBU getM2Pitch() const { return m2_pitch; }
    inline unsigned getNumLayers() const { return nLayers; }
    inline unsigned getSize(unsigned dimension) const { return (dimension ? ySize : xSize); }
    inline std::string getLayerName(int layerIndex) const { return layerNames[layerIndex]; }
    inline unsigned getLayerDirection(int layerIndex) const { return layerDirections[layerIndex]; }
    
    inline uint64_t hashCell(const GRPoint& point) const {
        return ((uint64_t)point.layerIdx * xSize + point.x) * ySize + point.y;
    };
    inline uint64_t hashCell(const int x, const int y) const { return (uint64_t)x * ySize + y; }
    inline DBU getGridline(const unsigned dimension, const int index) const { return gridlines[dimension][index]; }
    utils::BoxT<DBU> getCellBox(utils::PointT<int> point) const;
    utils::BoxT<int> rangeSearchCells(const utils::BoxT<DBU>& box) const;
    inline GraphEdge getEdge(const int layerIndex, const int x, const int y) const {return graphEdges[layerIndex][x][y]; }
    
    // Costs
    DBU getEdgeLength(unsigned direction, unsigned edgeIndex) const;
    CostT getWireCost(const int layerIndex, const utils::PointT<int> u, const utils::PointT<int> v) const;
    CostT getViaCost(const int layerIndex, const utils::PointT<int> loc) const;
    inline CostT getUnitViaCost() const { return unit_via_cost; }
    
    // Misc
    void selectAccessPoints(GRNet& net, robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>>& selectedAccessPoints) const;
    
    // Methods for updating demands 
    void commitTree(const std::shared_ptr<GRTreeNode>& tree, const bool reverse = false);
    void clean_gridGraph();//清理gridGraph里面所有graphedge的demand，将所有的totallength和所有的totalvianum清零，作用是把pretend stage1的已经commit的tree给清空掉
    
    // Checks
    inline bool checkOverflow(const int layerIndex, const int x, const int y) const { return getEdge(layerIndex, x, y).getResource() < 0.0; }
    int checkOverflow(const int layerIndex, const utils::PointT<int> u, const utils::PointT<int> v) const; // Check wire overflow
    int checkOverflow(const std::shared_ptr<GRTreeNode>& tree) const; // Check routing tree overflow (Only wires are checked)
    std::string getPythonString(const std::shared_ptr<GRTreeNode>& routingTree) const;
    
    // 2D maps
    void extractBlockageView(GridGraphView<bool>& view) const;
    void extractCongestionView(GridGraphView<bool>& view) const; // 2D overflow look-up table
    void extractEstimatedCongestionView_by_Steinertree(GridGraphView<bool>& view) const; // 2D estimated look-up table 使用steiner tree来实现
    void extractEstimatedCongestionView_by_RUDY(GridGraphView<bool>& view) const; // 2D estimated look-up table 使用RUDY来实现
    void extractWireCostView(GridGraphView<CostT>& view) const;
    void updateWireCostView(GridGraphView<CostT>& view, std::shared_ptr<GRTreeNode> routingTree) const;

    //CostT getdemand_for_mask(const int layerIndex, const utils::PointT<int> lower, const CapacityT demand = 1.0) const;
    CostT getWiredemand_for_mask(const int layerIndex, const utils::PointT<int> lower, const CapacityT demand = 1.0) const;
    float getViademand_for_mask(const int layerIndex, const utils::PointT<int> loc) const;
    
    // For visualization
    void write(const std::string heatmap_file="heatmap.txt") const;
    
private:
    const DBU libDBU;
    DBU m2_pitch;
    const Parameters& parameters;
    
    unsigned nLayers;
    unsigned xSize;
    unsigned ySize;
    vector<vector<DBU>> gridlines;
    vector<vector<DBU>> gridCenters;
    vector<std::string> layerNames;
    vector<unsigned> layerDirections;
    vector<DBU> layerMinLengths;
    
    // Unit costs
    CostT unit_length_wire_cost;
    CostT unit_via_cost;
    vector<CostT> unit_length_short_costs;
    
    DBU totalLength = 0;
    int totalNumVias = 0;
    vector<vector<vector<GraphEdge>>> graphEdges; 
    // gridEdges[l][x][y] stores the edge {(l, x, y), (l, x+1, y)} or {(l, x, y), (l, x, y+1)}
    // depending on the routing direction of the layer
    
    utils::IntervalT<int> rangeSearchGridlines(const unsigned dimension, const utils::IntervalT<DBU>& locInterval) const;
    // Find the gridlines within [locInterval.low, locInterval.high]
    utils::IntervalT<int> rangeSearchRows(const unsigned dimension, const utils::IntervalT<DBU>& locInterval) const;
    // Find the rows/columns overlapping with [locInterval.low, locInterval.high]
    
    // Utility functions for cost calculation
    inline CostT getUnitLengthWireCost() const { return unit_length_wire_cost; }
    // CostT getUnitViaCost() const { return unit_via_cost; }
    inline CostT getUnitLengthShortCost(const int layerIndex) const { return unit_length_short_costs[layerIndex]; }
    
    inline double logistic(const CapacityT& input, const double slope) const;
    CostT getWireCost(const int layerIndex, const utils::PointT<int> lower, const CapacityT demand = 1.0) const;
    
    
    // Methods for updating demands 
    void commit(const int layerIndex, const utils::PointT<int> lower, const CapacityT demand);
    void commitWire(const int layerIndex, const utils::PointT<int> lower, const bool reverse = false);
    void commitVia(const int layerIndex, const utils::PointT<int> loc, const bool reverse = false);
    
};

template <typename Type>
class GridGraphView: public vector<vector<vector<Type>>> {
public:
    // 检查两点是否连通
    bool check(const utils::PointT<int>& u, const utils::PointT<int>& v) const {
        //  assert 断言u.x == v.x || u.y == v.y，确保两点处于同一行或同一列
        assert(u.x == v.x || u.y == v.y);
        //  如果处于同一行，则遍历该行，如果有连通，则返回true
        if (u.y == v.y) {
            int l = min(u.x, v.x), h = max(u.x, v.x);
            for (int x = l; x < h; x++) {
                if ((*this)[MetalLayer::H][x][u.y]) return true;
            }
        //  如果处于同一列，则遍历该列，如果有连通，则返回true
        } else {
            int l = min(u.y, v.y), h = max(u.y, v.y);
            for (int y = l; y < h; y++) {
                if ((*this)[MetalLayer::V][u.x][y]) return true;
            }
        }
        return false;
    }
    
    // 求两点之间的和
    Type sum(const utils::PointT<int>& u, const utils::PointT<int>& v) const {
        //  assert 断言u.x == v.x || u.y == v.y，确保两点处于同一行或同一列
        assert(u.x == v.x || u.y == v.y);
        //  初始化和为0
        Type res = 0;
        // 如果处于同一行，则遍历该行，求和
        if (u.y == v.y) {
            int l = min(u.x, v.x), h = max(u.x, v.x);
            for (int x = l; x < h; x++) {
                res += (*this)[MetalLayer::H][x][u.y];
            }
        //  如果处于同一列，则遍历该列，求和
        } else {
            int l = min(u.y, v.y), h = max(u.y, v.y);
            for (int y = l; y < h; y++) {
                res += (*this)[MetalLayer::V][u.x][y];
            }
        }
        // 返回求和结果
        return res;
    }
};
