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

    // Get cap info
    std::ifstream cap_stream(params.cap_file);
    cap_stream >> n_layers >> size_x >> size_y;
    cap_stream >> unit_length_wire_cost >> unit_via_cost;
    unit_length_short_costs.resize(n_layers);
    horizontal_gcell_edge_lengths.resize(size_x);
    vertical_gcell_edge_lengths.resize(size_y);
    for(int i = 0; i < n_layers; i++)
        cap_stream >> unit_length_short_costs[i];
    for(int i = 0; i < size_x; i++)
        cap_stream >> horizontal_gcell_edge_lengths[i];
    for(int i = 0; i < size_y; i++)
        cap_stream >> vertical_gcell_edge_lengths[i];
    for (int z = 0; z < n_layers; z++)
    {
        std::string name;
        int direction;
        cap_stream >> name >> direction;
        layer_names.push_back(name);
        layer_directions.push_back(direction);

        std::vector<std::vector<CapacityT>> capacity(size_y, std::vector<CapacityT>(size_x));
        for (int y = 0; y < size_y; ++y)
            for (int x = 0; x < size_x; ++x)
                cap_stream >> capacity[y][x];
        gcell_edge_capaicty.push_back(std::move(capacity));
    }

    // Get net info
    std::ifstream net_stream(params.net_file);
    std::string line;
    std::string name;
    std::vector<std::vector<std::tuple<int, int, int>>> accessPoints;
    while (std::getline(net_stream, line))
    {
        auto it = std::find_if(line.begin(), line.end(), [](char c)
                               { return !std::isspace(c); });
        if (it == line.end())
            continue;
        else if (*it == '(') // access begin
        {
            // do nothing
        }
        else if (*it == ')') // access end
        {
            // commit a net
            net_names.push_back(name);
            net_access_points.push_back(accessPoints);
            accessPoints.clear();
        }
        else if (*it == '[') // access points
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
            accessPoints.push_back(std::move(access));
        }
        else // net name
        {
            name = line;
            if (name.back() == '\n')
                name.pop_back();
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
