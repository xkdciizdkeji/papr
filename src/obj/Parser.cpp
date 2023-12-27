#include "Parser.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cmath>
#include <set>
#include <sstream>
#include <algorithm>
#include <unordered_map>

void Parser::read(std::string cap_file, std::string PNet_file){
    log() << "parsing..." << std::endl;
    // Parse resource file
    // Get layout dimensions
    std::ifstream resourceFile(cap_file);

    std::string line;
    std::getline(resourceFile, line);
    std::vector<int> dimensions;
    std::istringstream iss(line);
    for (int value; iss >> value;) {
        dimensions.push_back(value);
    }
    int nLayers = dimensions[0];
    int xSize = dimensions[1];
    int ySize = dimensions[2];

    log() << "dimensions: " << nLayers << " " << xSize << " " << ySize << std::endl;

    // Get unit costs
    std::getline(resourceFile, line);
    std::istringstream unitCosts(line);
    // double unit_length_wire_cost, unit_via_cost;
    unitCosts >> unit_length_wire_cost >> unit_via_cost;

    // std::vector<double> unit_length_short_costs;
    double cost;
    while (unitCosts >> cost) {
        unit_length_short_costs.push_back(cost);
        // log() << cost<<std::endl;
    }
    log() << "Got unit costs: " << unit_length_wire_cost << " " << unit_via_cost << " "  << std::endl;


    // Get edge lengths
    DBU hcost, vcost;
    std::vector<DBU> hEdgeLengths;
    std::getline(resourceFile, line);
    std::istringstream hEdgeLine(line);
    DBU hcostTotal = 0;
    hEdgeLengths.push_back(hcostTotal);
    while (hEdgeLine >> hcost) {
        hcostTotal += hcost;
        hEdgeLengths.push_back(hcostTotal);
    }
    gridlines.push_back(hEdgeLengths);
    

    // for (int x=0; x<xSize; ++x) {
    //     log() << "hedge " << x << " length " << hEdgeLengths[x] << std::endl;
    // }

    std::vector<DBU> vEdgeLengths;
    std::getline(resourceFile, line);
    std::istringstream vEdgeLine(line);
    DBU vcostTotal = 0;
    vEdgeLengths.push_back(vcostTotal);
    while (vEdgeLine >> vcost) {
        vcostTotal += vcost;
        vEdgeLengths.push_back(vcostTotal);
    }
    gridlines.push_back(vEdgeLengths);
    // for (int y=0; y<ySize; ++y) {
    //     log() << "vedge " << y << " length " << vEdgeLengths[y] << std::endl;
    // }

    log() << "Got edge lengths: " << std::endl;

    // Get capacity map
    std::vector<std::vector<std::vector<CostT>>> Capacity(nLayers, std::vector<std::vector<double>>(xSize, std::vector<double>(ySize)));
    std::vector<int> layerDirections;
    std::vector<double> layerMinLengths;

    for (int l = 0; l < nLayers; ++l) {
        std::getline(resourceFile, line);
        std::istringstream layerInfo(line);
        std::string layer_name;
        layerInfo >> layer_name;
        int direction;
        layerInfo >> direction;
        layerDirections.push_back(direction);
        double min_length;
        layerInfo >> min_length;
        layerMinLengths.push_back(min_length);

        MetalLayer layer(layer_name, direction, min_length, l);
        layers.push_back(layer);

        // handle grid capicity
        for (int y = 0; y < ySize; ++y) {
            std::getline(resourceFile, line);
            std::istringstream capacityInfo(line);
            vector<DBU> rowCapicities;
            for (int x = 0; x < xSize; ++x) {
                capacityInfo >> Capacity[l][x][y];
            }
        }
    }
    CapacityMap = Capacity;

    // for(int l = 0; l < nLayers; ++l) {
    //     for (int x = 0; x < xSize; ++x) {
    //         for (int y = 0; y < ySize; ++y) {
    //             log() << "Capacity[" << l << "][" << x << "][" << y << "] = " << CapacityMap[l][x][y] << std::endl;
    //         }
    //     }
    // }
    log() << "Got capacity map" << std::endl;

    for (int l=0; l< nLayers; ++l) {
        log() << "Layer " << l << " direction " << layerDirections[l] << " minLength " << layerMinLengths[l] << std::endl;
    }

    // Parse PNet file
    // Get PNet info
    std::string name;
    int numPins = 0;
    // std::vector<PNet> PNets;
    // std::unordered_map<std::string, PNet> PNets;
    // int PNetId = -1;
    std::vector<std::vector<std::vector<int>>> accessPoints;

    std::ifstream PNetFile(PNet_file);

    int64_t PNetId = 0;
    while (std::getline(PNetFile, line)) {
        // if (line.find("PNet") != std::string::npos || line.find("pin") != std::string::npos) {
        if (line.find("(") == std::string::npos && line.find(")") == std::string::npos && line.length()>1) {
            // name = line.substr(0, line.size() - 1);
            name = line;
            size_t found = name.find('\n');
            if (found != std::string::npos) {
                name.erase(found, 1);
            }
            // PNetId = std::stoi(line.substr(3));
            numPins = 0;
            // log() << name << " " << PNetId << std::endl;
        } else if (line.find('[') != std::string::npos) {
            std::vector<std::vector<int>> access;
            std::string text = line.substr(1, line.size() - 2); // Remove brackets and trailing comma
            std::string charsToRemove = "(),";
            text.erase(std::remove_if(text.begin(), text.end(), [&charsToRemove](char c) {
                return charsToRemove.find(c) != std::string::npos;
            }), text.end());
            // log() << "current line is: " << text << std::endl;
            std::istringstream ss(text);
            int x, y, z;
            while (ss >> x >> y >> z) {
                std::vector<int> point;
                point.push_back(x);
                point.push_back(y);
                point.push_back(z);
                access.push_back(point);
            }
            accessPoints.push_back(access);
            numPins++;
        } else if (line.find(')') != std::string::npos) {
            // if (PNetId == 3153 || PNetId == 3151) {
            //     log() << name << " " << PNetId << " " << numPins << std::endl;
            // }
            // PNets.push_back(PNet(name, numPins, accessPoints));
            PNet PNet(PNetId, name, numPins, accessPoints);
            PNetId++;
            nets.push_back(PNet);
            accessPoints.clear();
        }
    }
    // for(int i = 0; i < nets.size(); ++i) {
    //     log() << "PNet " << i << " " << nets[i].name << " " << nets[i].numPins << std::endl;
    //     for(int j = 0; j < nets[i].numPins; ++j) {
    //         for(int k = 0; k < nets[i].accessPoints[j].size(); ++k) {
    //             log() << nets[i].accessPoints[j][k][0] << " " << nets[i].accessPoints[j][k][1] << " " << nets[i].accessPoints[j][k][2] << std::endl;
    //         }
    //     }
    // }

    log() << "Got PNet info" << std::endl;
    log() << "Finished reading cap/net" << std::endl;
    logmem();
    logeol();
    
    log() << "design statistics" << std::endl;
    loghline();
    log() << "num of nets :        " << nets.size() << std::endl;
    log() << "gcell grid:          " << gridlines[0].size() - 1 << " x " << gridlines[1].size() - 1 << " x " << getNumLayers() << std::endl;
    log() << "unit length wire:    " << unit_length_wire_cost << std::endl;
    log() << "unit via:            " << unit_via_cost << std::endl;
    logeol();

}