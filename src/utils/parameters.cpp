#include "parameters.h"
#include "utils.h"

Parameters::Parameters(int argc, const char *argv[])
{
    if (argc <= 1)
    {
        log() << "Too few args..." << std::endl;
        exit(1);
    }
    cap_file = argv[1];
    net_file = argv[2];
    out_file = argv[3];
    // threads = std::stoi(argv[4]);
    log() << "cap file: " << cap_file << std::endl;
    log() << "net file: " << net_file << std::endl;
    log() << "output  : " << out_file << std::endl;
    log() << "threads : " << threads << std::endl;
    log() << std::endl;
}
