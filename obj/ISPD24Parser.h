#pragma once
#include <vector>
#include <string>
#include "MetalLayer.h"
#include "../utils/utils.h"

struct ISPD24Parser
{
    ISPD24Parser(const Parameters &params);

    unsigned int n_layers, size_x, size_y;
    CostT unit_length_wire_cost;
    CostT unit_via_cost;
    std::vector<CostT> unit_length_short_costs;
    std::vector<DBU> horizontal_gcell_edge_lengths;
    std::vector<DBU> vertical_gcell_edge_lengths;
    std::vector<std::string> layer_names;
    std::vector<unsigned int> layer_directions;
    std::vector<DBU> layer_min_lengths;
    std::vector<std::vector<std::vector<CapacityT>>> gcell_edge_capaicty; // [layerIdx][y][x]
    std::vector<std::string> net_names;
    std::vector<std::vector<std::vector<std::tuple<int, int, int>>>> net_access_points; // [netId][pinId][accessPoint]
};