#include "global.h"
#include "obj/Design.h"
#include "obj/Parser.h"
#include "gr/GlobalRouter.h"

int main(int argc, char* argv[]) {
    // 
    // input lef/def
    // 
    // logeol(2);
    // log() << "GLOBAL ROUTER CUGR" << std::endl;
    // logeol(2);
    // // Parse parameters
    // Parameters parameters(argc, argv);
    
    // // Read LEF/DEF
    // Design design(parameters);
    
    // // Global router
    // GlobalRouter globalRouter(design, parameters);
    // // globalRouter.route();
    // globalRouter.netSortRoute(10);
    // globalRouter.write();
    
    // logeol();
    // log() << "Terminated." << std::endl;
    // loghline();
    // logmem();
    // logeol();

    // 
    // input cap/net
    // 
    logeol(2);
    log() << "GLOBAL ROUTER CUGR" << std::endl;
    logeol(2);
    // Parse parameters
    ParametersISPD24 ParametersISPD24(argc, argv);
    
    // Read CAP/NET
    Parser parser(ParametersISPD24);
    
    // Global router
    GlobalRouter globalRouter(parser, ParametersISPD24);
    globalRouter.route();
    // globalRouter.netSortRoute(10);
    globalRouter.write();
    
    logeol();
    log() << "Terminated." << std::endl;
    loghline();
    logmem();
    logeol();
}