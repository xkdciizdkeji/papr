#pragma once
#include "global.h"
#include "GeoTypes.h"
#include "Layers.h"
#include "Net.h"
#include "Instance.h"

using CostT = double;

class PNet {
public:
    std::string name;
    std::int64_t id;
    int numPins;
    std::vector<std::vector<std::vector<int>>> accessPoints;

    PNet(std::int64_t idx,std::string name, int num, std::vector<std::vector<std::vector<int>>>& access) :id(idx), name(name), numPins(num), accessPoints(access) {};
    PNet() {};
};

class Parser{
    public:
    Parser(const ParametersISPD24& params): parametersISPD24(params) {
        read(parametersISPD24.cap_file, parametersISPD24.net_file);
        // setUnitCosts();
    }

    CostT getUnitLengthWireCost() const { return unit_length_wire_cost; }
    CostT getUnitViaCost() const { return unit_via_cost; }
    CostT getUnitLengthShortCost(const int layerIndex) const { return unit_length_short_costs[layerIndex]; }
    
    int getNumLayers() const { return layers.size(); }
    const MetalLayer& getLayer(int layerIndex) const { return layers[layerIndex]; }
    // void getPinShapes(const PinReference& pinRef, vector<BoxOnLayer>& pinShapes) const;
    
    // For global routing 
    const vector<vector<DBU>>& getGridlines() const { return gridlines; }
    const vector<vector<vector<CostT>>>& getCapacityMap() const { return CapacityMap; }
    const vector<PNet>& getAllNets() const { return nets; }
    // void getAllObstacles(vector<vector<utils::BoxT<DBU>>>& allObstacles, bool skipM1 = true) const;
    
private:
    const ParametersISPD24& parametersISPD24;
    
    // utils::BoxT<DBU> dieRegion;
    vector<MetalLayer> layers;
    // vector<Macro> macros; // macros with a SPECIFIC orientation
    // vector<Instance> instances;
    vector<PNet> nets;
    // vector<BoxOnLayer> obstacles;
    
    // For detailed routing
    CostT unit_length_wire_cost;
    CostT unit_via_cost;
    vector<CostT> unit_length_short_costs;
    
    // For global routing
    const static int defaultGridlineSpacing = 3000;
    vector<vector<DBU>> gridlines;
    std::vector<std::vector<std::vector<CostT>>> CapacityMap;
    
    void read(std::string cap_file, std::string net_file);
    // void setUnitCosts();

};