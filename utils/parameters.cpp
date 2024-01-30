#include "parameters.h"
#include "utils.h"

Parameters::Parameters(int argc, const char *argv[])
{
    if (argc <= 1)
    {
        log() << "Too few args..." << std::endl;
        exit(1);
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-cap") == 0)
        {
            cap_file = argv[++i];
        }
        else if (strcmp(argv[i], "-net") == 0)
        {
            net_file = argv[++i];
        }
        else if (strcmp(argv[i], "-output") == 0)
        {
            out_file = argv[++i];
        }
        else if (strcmp(argv[i], "-threads") == 0)
        {
            threads = std::stoi(argv[++i]);
        }
        else
        {
            log() << "Unrecognized arg..." << std::endl;
            log() << argv[i] << std::endl;
        }
    }
    // threads = std::stoi(argv[4]);
    log() << "cap file: " << cap_file << std::endl;
    log() << "net file: " << net_file << std::endl;
    log() << "output  : " << out_file << std::endl;
    log() << "threads : " << threads << std::endl;
    log() << std::endl;
}
