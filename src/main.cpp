#include "global.h"
#include "obj/ISPD24Parser.h"
#include "utils/utils.h"
#include "gr/GlobalRouter.h"

int main(int argc, const char *argv[])
{
    logeol(2);
    log() << "GLOBAL ROUTER CUGR" << std::endl;
    logeol(2);
    // Parse parameters
    Parameters parameters(argc, argv);

    // Read CAP/NET
    ISPD24Parser parser(parameters);

    // Global router
    GlobalRouter globalRouter(parser, parameters);
    globalRouter.route();
    //globalRouter.write();

    logeol();
    log() << "Terminated." << std::endl;
    loghline();
    logmem();
    logeol();
}

