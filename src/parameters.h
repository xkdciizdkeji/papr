#include <string>

struct Parameters
{
    std::string lef_file;
    std::string def_file;
    std::string out_file;
    int threads = 1;
    //
    const double weight_wire_length = 0.5;
    const double weight_via_number = 4.0;
    const double weight_short_area = 500.0;
    //
    const int min_routing_layer = 1;
    const double cost_logistic_slope = 1.0;
    const double max_detour_ratio = 0.25; // allowed stem length increase to trunk length ratio
    const int target_detour_count = 20;
    const double via_multiplier = 2.0;
    //
    const double maze_logistic_slope = 0.5;
    //
    const double pin_patch_threshold = 20.0;
    const int pin_patch_padding = 1;
    const double wire_patch_threshold = 2.0;
    const double wire_patch_inflation_rate = 1.2;
    //
    const bool write_heatmap = false;

    Parameters(int argc, const char *argv[]);
};
