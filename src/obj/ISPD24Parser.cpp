#include "ISPD24Parser.h"
#include "../utils/log.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace utils;

ISPD24Parser::ISPD24Parser(const Parameters &params)
{
    log() << "parsing..." << std::endl;
    std::string line;

    // Parser resource file
    std::ifstream cap_stream(params.cap_file);
    // Get layout dimensions
    std::getline(cap_stream, line);
    std::istringstream dimension_line(line);
    dimension_line >> n_layers >> size_x >> size_y;
    // Get unit costs
    std::getline(cap_stream, line);
    std::istringstream unit_costs_line(line);
    unit_costs_line >> unit_length_wire_cost >> unit_via_cost;
    unit_length_short_costs.reserve(n_layers);
    for(CostT cost; unit_costs_line >> cost;)
        unit_length_short_costs.push_back(cost);
    // Get edge lengths
    std::getline(cap_stream, line);
    std::istringstream h_edge_line(line);
    horizontal_gcell_edge_lengths.reserve(size_x - 1);
    for(DBU hl; h_edge_line >> hl;)
        horizontal_gcell_edge_lengths.push_back(hl);
    std::getline(cap_stream, line);
    std::istringstream v_edge_line(line);
    vertical_gcell_edge_lengths.reserve(size_y - 1);
    for(DBU vl; v_edge_line >> vl;)
        vertical_gcell_edge_lengths.push_back(vl);
    // Get capacity map
    for (int z = 0; z < n_layers; z++)
    {
        std::string name;
        unsigned int direction;
        DBU min_length;
        std::getline(cap_stream, line);
        std::istringstream layer_info_line(line);
        layer_info_line >> name >> direction >> min_length;
        layer_names.push_back(name);
        layer_directions.push_back(direction);
        layer_min_lengths.push_back(min_length);

        std::vector<std::vector<CapacityT>> capacity(size_y, std::vector<CapacityT>(size_x));
        for (int y = 0; y < size_y; ++y)
        {
            std::getline(cap_stream, line);
            std::istringstream capacity_line(line);
            for (int x = 0; x < size_x; ++x)
                capacity_line >> capacity[y][x];
        }
        gcell_edge_capaicty.push_back(std::move(capacity));
    }

    // Get net info
    std::ifstream net_stream(params.net_file);
    std::string name;
    std::vector<std::vector<std::tuple<int, int, int>>> accessPoints;
    while (std::getline(net_stream, line))
    {
        if(line.find('(') == std::string::npos && line.find(')') == std::string::npos && line.length() > 1)
        {
            name = line;
            if(name.back() == '\n')
                name.pop_back();
        }
        else if(line.find('[') != std::string::npos)
        {
            auto is_useless = [](char c)
            {
                return c == '[' || c == ']' || c == ',' || c == '(' || c == ')';
            };
            std::replace_if(line.begin(), line.end(), is_useless, ' ');
            std::istringstream iss(line);
            std::vector<std::tuple<int, int, int>> access;
            int x, y, z;
            while(iss >> z >> x >> y)
                access.emplace_back(x, y, z);
            accessPoints.push_back(access);
        }
        else if (line.find(')') != std::string::npos)
        {
            net_names.push_back(name);
            net_access_points.push_back(accessPoints);
            accessPoints.clear();
        }
    }

    log() << "Finished parsing\n";
    logmem();
    logeol();

    log() << "design statistics\n";
    loghline();
    log() << "num of nets :        " << net_names.size() << "\n";
    log() << "gcell grid:          " << size_x << " x " << size_y << " x " << n_layers << "\n";
    log() << "unit length wire:    " << unit_length_wire_cost << "\n";
    log() << "unit via:            " << unit_via_cost << "\n";
    logeol();
}
