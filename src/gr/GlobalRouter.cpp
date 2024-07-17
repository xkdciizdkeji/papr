#include "GlobalRouter.h"
#include "PatternRoute.h"
#include "MazeRoute.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <fstream>

#include "Torchroute.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ops/full_native.h>
#include <torch/cuda.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <c10/cuda/CUDAStream.h>


//配置libtorch所需操作：
// CMakeLists里面，搜torch几个字母，凡是涉及到文件路径的都得改
// 在.vscode文件夹里面的c_cpp_properties.json文件里面涉及到库的路径的都得改
// 一定要注意下libtorch一定要选linux，我不小心点快了选了windows，搞了好久才注意到包是有问题的


using namespace std;
using std::vector;
using std::max;
using std::min;

GlobalRouter::GlobalRouter(const Design& design, const Parameters& params): 
    gridGraph(design, params), parameters(params) {
    // Instantiate the global routing netlist
    const vector<Net>& baseNets = design.getAllNets();
    nets.reserve(baseNets.size());
    for (const Net& baseNet : baseNets) {
        nets.emplace_back(baseNet, design, gridGraph);
    }
}

void GlobalRouter::route() {
    int n1 = 0, n2 = 0, n3 = 0;
    double t1 = 0, t2 = 0, t3 = 0;
    
    auto t = std::chrono::high_resolution_clock::now();
    
    vector<int> netIndices;
    
    netIndices.reserve(nets.size());
    for (const auto& net : nets) netIndices.push_back(net.getIndex());
    // Stage 1: Pattern routing
    n1 = netIndices.size();
    PatternRoute::readFluteLUT();
    log() << "stage 1: pattern routing" << std::endl;
    sortNetIndices(netIndices);
    std::unordered_map<int, PatternRoute> PatternRoutes;
    std::unordered_map<int, PatternRoute> PatternRoutes_second;//记录第二个tree
    std::unordered_map<int, PatternRoute> PatternRoutes_stage2;
    int net_num=nets.size();
    int Max_tree_num_in_single_net=2;
    std::vector<std::unordered_map<int, PatternRoute>> PatternRoutes_multi(Max_tree_num_in_single_net);
    //Modified by IrisLin&Feng
    //vector<vector<vector<TorchEdge>>> TorchEdges(netIndices.size(),vector<vector<TorchEdge>>(1));//<net<tree<twopinnet>>>
    //vector<TorchEdge> TorchEdges0;
    vector<vector<vector<TorchEdge>>> TorchEdges0(nets.size(),vector<vector<TorchEdge>>(Max_tree_num_in_single_net));//TorchEdges0.reserve(nets.size());
    
    vector<TorchEdge> TorchEdges;
    int batch_size=10000;
    vector<vector<TorchEdge>> TorchEdges_batch(ceil(float(net_num)/batch_size));//每batch_size个net一组
    //vector<vector<bool>> enter_pretend_stage2_flag(nets.size(),vector<bool>(Max_tree_num_in_single_net,false));
    
    
    torch::Tensor link_P_tree_to_P_pattern;
    vector<int> link_P_tree_to_P_pattern_location_dimension0;
    vector<int> link_P_tree_to_P_pattern_location_dimension1;

    vector<torch::Tensor> link_P_tree_to_P_pattern_batch(ceil(float(net_num)/batch_size));
    vector<vector<int>> link_P_tree_to_P_pattern_location_dimension0_batch(ceil(float(net_num)/batch_size));
    vector<vector<int>> link_P_tree_to_P_pattern_location_dimension1_batch(ceil(float(net_num)/batch_size));

    torch::Tensor demand_mask;
    torch::Tensor wirelength_mask;
    torch::Tensor viacount_mask;
    
    
    
    //一些不变量
    vector<vector<vector<int>>> pattern_selection(net_num,vector<vector<int>>(Max_tree_num_in_single_net));
    vector<int> tree_selection_vector(net_num,0);
    vector<int> tree_next_update_vector(net_num,1);
    int two_pin_net_count=0;
    int net_count=0;
    int gcell_num_x = static_cast<int>(gridGraph.getSize(0));
    int gcell_num_y = static_cast<int>(gridGraph.getSize(1));

    // //optimize here
    // c10::DeviceType dev;
    // //c10::Device device;
    
    // log()<<"start_get_optim_device_function"<<std::endl;
    // get_optim_device(&dev);//如果cuda没问题，那么会输出current optimization device: cuda
    c10::Device dev(torch::kCUDA, 1);
    


    std::vector<float> compresed_capacitymap(2*gcell_num_x*gcell_num_y);//layer,x,y的顺序
    std::cout<< gridGraph.getNumLayers()<<std::endl;
    for (int y = 0; y < gridGraph.getSize(1); y++)
    {
        for (int x = 0; x < gridGraph.getSize(0); x++)
        {
            //for (int hlayer=0 ; hlayer<gridGraph.getNumLayers(); hlayer+=2) compresed_capacitymap[0][x][y]+=GlobalRouter::capacitymap[hlayer][y][x];
            for(int hlayer=2 ; hlayer<gridGraph.getNumLayers(); hlayer+=2) compresed_capacitymap[0*gcell_num_x*gcell_num_y+x*gcell_num_y+y] += gridGraph.getEdge(hlayer,x,y).capacity;
            //for (int vlayer=1 ; vlayer<gridGraph.getNumLayers(); vlayer+=2) compresed_capacitymap[1][x][y]+=GlobalRouter::capacitymap[vlayer][y][x];
            for(int vlayer=1 ; vlayer<gridGraph.getNumLayers(); vlayer+=2) compresed_capacitymap[1*gcell_num_x*gcell_num_y+x*gcell_num_y+y] += gridGraph.getEdge(vlayer,x,y).capacity;
        }
        
    }
    //auto compresed_capacity_map_tensor = torch::tensor(compresed_capacitymap,torch::kFloat).reshape({2,gcell_num_x,gcell_num_y}).to(dev);
    auto compresed_capacity_map_tensor = torch::tensor(compresed_capacitymap,torch::kFloat).reshape({2*gcell_num_x*gcell_num_y,1}).to(dev);
    log()<<"2d_capacitymap_finish"<<std::endl;
    logeol();



    for (const int netIndex : netIndices) {
        vector<PatternRoute> patternRoute_multi(Max_tree_num_in_single_net,PatternRoute(nets[netIndex], gridGraph, parameters));
        patternRoute_multi[0].constructSteinerTree();
        PatternRoutes_multi[0].insert(std::make_pair(netIndex, patternRoute_multi[0]));
        auto torch_edge = patternRoute_multi[0].getsT()->getTorchEdges(patternRoute_multi[0].getsT());//获取该net的twopinnet
        TorchEdges0[netIndex][0].insert(TorchEdges0[netIndex][0].end(),torch_edge.begin(),torch_edge.end());
        
        
        for (int tree_num=1;tree_num<Max_tree_num_in_single_net;tree_num++){
            patternRoute_multi[tree_num].constructSteinerTree_Random();
            PatternRoutes_multi[tree_num].insert(std::make_pair(netIndex, patternRoute_multi[tree_num]));
            auto Torchedge = patternRoute_multi[tree_num].getsT()->getTorchEdges(patternRoute_multi[tree_num].getsT());//获取该net的twopinnet
            TorchEdges0[netIndex][tree_num].insert(TorchEdges0[netIndex][tree_num].end(),Torchedge.begin(),Torchedge.end());
        }
        //std::cout<<patternRoute_multi[0].getsT()->getPythonString(patternRoute_multi[0].getsT())<<std::endl;
        patternRoute_multi[0].constructRoutingDAG();
        patternRoute_multi[0].run();
        gridGraph.commitTree(nets[netIndex].getRoutingTree());


        
        
        
        
        // PatternRoute patternRoute_second(nets[netIndex], gridGraph, parameters);
        // patternRoute_second.constructSteinerTree_Random();
        // //PatternRoutes.insert(std::make_pair(netIndex, patternRoute));
        // PatternRoutes_second.insert(std::make_pair(netIndex, patternRoute_second));


        // auto Torchedge_second = patternRoute_second.getsT()->getTorchEdges(patternRoute_second.getsT());//获取该net的twopinnet
        // //TorchEdges0.insert(TorchEdges0.end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        // //TorchEdges0[netIndex][0].insert(TorchEdges0[netIndex][0].end(),Torchedge.begin(),Torchedge.end());
        // TorchEdges0[netIndex][1].insert(TorchEdges0[netIndex][1].end(),Torchedge_second.begin(),Torchedge_second.end());

        // PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
        // patternRoute.constructSteinerTree();
        // PatternRoutes.insert(std::make_pair(netIndex, patternRoute));
        // auto Torchedge = patternRoute.getsT()->getTorchEdges(patternRoute.getsT());//获取该net的twopinnet
        
        // TorchEdges0[netIndex][0].insert(TorchEdges0[netIndex][0].end(),Torchedge.begin(),Torchedge.end());
        // //TorchEdges0[netIndex][1]=TorchEdges0[netIndex][0];
        // patternRoute.constructRoutingDAG();
        // // PatternRoutes_second.insert(std::make_pair(netIndex, patternRoute));
        // patternRoute.run();
        // gridGraph.commitTree(nets[netIndex].getRoutingTree());
        // //link_P_tree_to_P_pattern_location_dimension0.insert(link_P_tree_to_P_pattern_location_dimension0.end(),TorchEdges.size()-Torchedge.size(),TorchEdges.size());
        // //link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net);
        // //std::cout<<"net_count"<<net_count<<std::endl;
        // //net_count++;
        // //link_P_tree_to_P_pattern_location_dimension1末尾增加net_count个Torchedge.size()
        // // TorchEdges[netIndex].resize(1);
        // // TorchEdges[netIndex][0].insert(TorchEdges[netIndex][0].end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        
    }
    log()<<"generate_all_two_pin_net_finish"<<std::endl;

    printStatistics();

    for(int dynamic_update=0;dynamic_update<1;dynamic_update++){
        netIndices.clear();
        for (const auto& net : nets) {
            if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
                netIndices.push_back(net.getIndex());
            }
        }
        log() << netIndices.size() << " overflows." << std::endl;
        logeol();

        n2 = netIndices.size();
        if (netIndices.size() > 0) {
            //log() << "stage 2: pattern routing with possible detours" << std::endl;
            log() << "pretend stage 2: in order to generate tree forest" << std::endl;



            GridGraphView<bool> congestionView; // (2d) direction -> x -> y -> has overflow?
            gridGraph.extractCongestionView(congestionView);
            // for (const int netIndex : netIndices) {
            //     GRNet& net = nets[netIndex];
            //     gridGraph.commitTree(net.getRoutingTree(), true);
            // }
            vector<double> time_ps2(3,0);
            sortNetIndices(netIndices);
            for (const int netIndex : netIndices) {
                GRNet& net = nets[netIndex];
                gridGraph.commitTree(net.getRoutingTree(), true);//true的目的是取消掉已经提交的tree，因为commit的本质就是将tree所占用的demand给更新的gridgraph上，那取消就是把已经占用的demand给减掉就相当于取消commit了

                PatternRoute patternRoute(net, gridGraph, parameters);
                //patternRoute.constructSteinerTree();
                auto selectedAccessPoints=patternRoute.constructSteinerTree_return_selectedAccessPoints();
                patternRoute.constructRoutingDAG();
                

                // auto patternRoute = PatternRoutes_second.find(netIndex)->second;
                // std::cout<<netIndex<<std::endl;
                // std::cout<<patternRoute.getsT()->getPythonString(patternRoute.getsT())<<std::endl;
                //std::cout<<"before_detour"<<std::endl;
                //string a=patternRoute.getrT()->getPythonString(patternRoute.getrT());
                auto t0=std::chrono::high_resolution_clock::now();
                patternRoute.constructDetours(congestionView); // KEY DIFFERENCE compared to stage 1
                time_ps2[0]+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                auto t1=std::chrono::high_resolution_clock::now();
                //std::cout<<"after_detour"<<std::endl;
                //string b=patternRoute.getrT()->getPythonString(patternRoute.getrT());
                //if(a!=b) std::cout<<"before_detour"<<std::endl<<a<<std::endl<<"after_detour"<<std::endl<<b<<std::endl;
                patternRoute.run();
                time_ps2[1]+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();
                auto t2=std::chrono::high_resolution_clock::now();
                patternRoute.clear_steinerTree();
                patternRoute.constructSteinerTree_based_on_routingTree(net.getRoutingTree(),selectedAccessPoints);
                time_ps2[2]+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t2).count();
                //patternRoute.constructSteinerTree_based_on_routingTree63(net.getRoutingTree());
                // std::cout<<patternRoute.getsT()->getPythonString(patternRoute.getsT())<<std::endl;
                //std::cout<<std::endl;
                patternRoute.clear_routingDag();
                // if (tree_selection_vector[netIndex]==0) PatternRoutes_second.insert(std::make_pair(netIndex, patternRoute));
                // else PatternRoutes.insert(std::make_pair(netIndex, patternRoute));
                // if (tree_selection_vector[netIndex]==0) {PatternRoutes_multi[1].erase(netIndex);PatternRoutes_multi[1].insert(std::make_pair(netIndex, patternRoute));}
                // else {PatternRoutes_multi[0].erase(netIndex);PatternRoutes_multi[0].insert(std::make_pair(netIndex, patternRoute));}

                //PatternRoutes_multi[tree_next_update_vector[netIndex]].erase(netIndex);
                // int a=PatternRoutes_multi[tree_next_update_vector[netIndex]].erase(netIndex);
                // std::cout<<(a?"yes":"no")<<std::endl;

                PatternRoutes_multi[tree_next_update_vector[netIndex]].erase(netIndex);
                PatternRoutes_multi[tree_next_update_vector[netIndex]].insert(std::make_pair(netIndex, patternRoute));
                
                //auto Torchedge = net.getTorchEdges(net.getRoutingTree());
                auto Torchedge = patternRoute.getsT()->getTorchEdges(patternRoute.getsT());//获取该net的twopinnet
                //TorchEdges0[netIndex][!tree_selection_vector[netIndex]].assign(Torchedge.begin(),Torchedge.end());
                //TorchEdges0[netIndex][tree_next_update_vector[netIndex]].assign(Torchedge.begin(),Torchedge.end());
                TorchEdges0[netIndex][tree_next_update_vector[netIndex]]=Torchedge;
                //enter_pretend_stage2_flag[netIndex][!tree_selection_vector[netIndex]]=true;
                //TorchEdges0[netIndex][1].clear();
                //TorchEdges0[netIndex][1].insert(TorchEdges0[netIndex][0].end(),Torchedge.begin(),Torchedge.end());
                // gridGraph.commitTree(net.getRoutingTree());
                // PatternRoutes_second.insert(std::make_pair(netIndex, patternRoute));
            }
            std::cout<<"time_ps2:"<<time_ps2<<std::endl;




            log()<<"generate_treeforest_finish"<<std::endl;
            //std::cout<<PatternRoutes.find(90169)
            
            // netIndices.clear();
            // for (const auto& net : nets) {
            //     if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
            //         netIndices.push_back(net.getIndex());
            //     }
            // }
            // //log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
            // log() << netIndices.size() << " overflows." << std::endl;
            // logeol();
            
        }

        // gridGraph.clean_gridGraph();//需要将已经占用的demand先清零一下
        // link_P_tree_to_P_pattern_location_dimension1.clear();
        // std::cout<<"link_P_tree_to_P_pattern_location_dimension1.size()"<<link_P_tree_to_P_pattern_location_dimension1.size()<<std::endl;
        // TorchEdges.clear();
        // //for(const auto& vec2D : TorchEdges0) for(const auto& vec1D : vec2D) TorchEdges.insert(TorchEdges.end(),vec1D.begin(),vec1D.end());
        // int tree_count=0;
        // for(const auto& vec2D : TorchEdges0){
        //     for(const auto& vec1D : vec2D){
        //         TorchEdges.insert(TorchEdges.end(),vec1D.begin(),vec1D.end());
        //         //link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),vec1D.size(),tree_count);
        //         link_P_tree_to_P_pattern_location_dimension1.resize(link_P_tree_to_P_pattern_location_dimension1.size()+vec1D.size(),tree_count);
        //         tree_count++;
            
        //     }
        // }


        gridGraph.clean_gridGraph();//需要将已经占用的demand先清零一下
        link_P_tree_to_P_pattern_location_dimension1_batch.clear();
        link_P_tree_to_P_pattern_location_dimension1_batch.resize(ceil(float(net_num)/batch_size));
        std::cout<<"link_P_tree_to_P_pattern_location_dimension1.size()"<<link_P_tree_to_P_pattern_location_dimension1.size()<<std::endl;
        TorchEdges_batch.clear();
        TorchEdges_batch.resize(ceil(float(net_num)/batch_size));
        std::cout<<ceil(float(net_num)/batch_size)<<std::endl;
        //std::cout<<TorchEdges_batch[ceil(net_num/batch_size)-1]<<std::endl;
        //for(const auto& vec2D : TorchEdges0) for(const auto& vec1D : vec2D) TorchEdges.insert(TorchEdges.end(),vec1D.begin(),vec1D.end());
        int tree_count_within_batch=0;
        int batch_index=0;
        for(int net_index=0;net_index<net_num;net_index++){
            const auto& vector2D=TorchEdges0[net_index];
            for(int tree_index=0;tree_index<Max_tree_num_in_single_net;tree_index++){
                const auto& vector1D=vector2D[tree_index];
                // std::cout<<"1"<<std::endl;
                TorchEdges_batch[batch_index].insert(TorchEdges_batch[batch_index].end(),vector1D.begin(),vector1D.end());
                link_P_tree_to_P_pattern_location_dimension1_batch[batch_index].resize(link_P_tree_to_P_pattern_location_dimension1_batch[batch_index].size()+vector1D.size(),tree_count_within_batch);
                tree_count_within_batch++;
            }
            // if ((net_index+1)%batch_size==0&&net_index!=0) {
            if ((net_index+1)%batch_size==0) {
                batch_index++;
                // std::cout<<batch_index<<std::endl;
                tree_count_within_batch=0;
            }
        }


        // for(int batch_index=0;batch_index<ceil(net_num/batch_size);batch_index++){
        //     for(int net_index=0;net_index<net_num;net_index++){
        //         const auto& vector2D=TorchEdges0[net_index];
        //         for(int tree_index=0;tree_index<Max_tree_num_in_single_net;tree_index++){
        //             const auto& vector1D=vector2D[tree_index];
        //             // std::cout<<"1"<<std::endl;
        //             TorchEdges_batch[batch_index].insert(TorchEdges_batch[batch_index].end(),vector1D.begin(),vector1D.end());
        //             link_P_tree_to_P_pattern_location_dimension1_batch[batch_index].resize(link_P_tree_to_P_pattern_location_dimension1_batch[batch_index].size()+vector1D.size(),tree_count_within_batch);
        //             tree_count_within_batch++;
        //         }
        //         // if ((net_index+1)%batch_size==0&&net_index!=0) {
        //         if ((net_index+1)%batch_size==0) {
        //             // std::cout<<"enter"<<std::endl;
        //             // batch_index++;
        //             tree_count_within_batch=0;
        //         }
        //     }
        // }
        std::cout<<"finish"<<std::endl;

        // log()<<"generate_all_two_pin_net_finish"<<std::endl;
        // logeol();
        // auto estimated_congestion_map_by_steinertree = generate_2d_estimated_congestion_map_by_steinertree(gridGraph.getSize(0),gridGraph.getSize(1),TorchEdges0);
        // log()<<"generate_2d_estimated_congestion_map_by_steinertree_finish"<<std::endl;
        // logeol();
        // auto estimated_congestion_map_by_RUDY = generate_2d_estimated_congestion_map_by_RUDY(1,1,gridGraph.getSize(0),gridGraph.getSize(1),nets);
        
        
        // for (const int netIndex : netIndices) {//这一次是为了建立完整的treeforest
        //     PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
        //     patternRoute.constructSteinerTree();
        //     //std::cout << "1" << std::endl;
        //     auto Torchedge = patternRoute.getsT()->getTorchEdges(patternRoute.getsT());//获取该net的twopinnet
        //     TorchEdges.insert(TorchEdges.end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        //     link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net);
        //     //std::cout << "2" << std::endl;
        //     if(check_Tree_Overflow_2D_estimated_congestion_map(Torchedge,estimated_congestion_map_by_steinertree)) {
        //         //std::cout << "3" << std::endl;
        //         patternRoute.constructRoutingDAG();
        //         //std::cout << "4" << std::endl;
        //         patternRoute.constructDetours(estimated_congestion_map_by_steinertree);//这个函数返回的类型不太对
        //         //std::cout << "5" << std::endl;
                
        //         auto Torchedge = patternRoute.getsT()->getTorchEdges(patternRoute.getsT());//获取该net的twopinnet
        //         //std::cout << "6" << std::endl;
        //         TorchEdges.insert(TorchEdges.end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        //         link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net+1);
        //     }
        //     else{
        //         //std::cout << "7" << std::endl;
        //         TorchEdges.insert(TorchEdges.end(),Torchedge.begin(),Torchedge.end());//对于没有overflow的tree，则再次加入twopinnet（也就是两个一样的tree）
        //         //std::cout << "8" << std::endl;
        //         link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net+1);
        //     }
        //     if(check_Tree_Overflow_2D_estimated_congestion_map(Torchedge,estimated_congestion_map_by_RUDY)){
        //         patternRoute.constructRoutingDAG();
        //         patternRoute.constructDetours(estimated_congestion_map_by_RUDY);
        //         auto Torchedge = patternRoute.getsT()->getTorchEdges(patternRoute.getsT());//获取该net的twopinnet
        //         TorchEdges.insert(TorchEdges.end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        //         link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net+2);
        //     }
        //     else{
        //         TorchEdges.insert(TorchEdges.end(),Torchedge.begin(),Torchedge.end());//对于没有overflow的tree，则再次加入twopinnet（也就是两个一样的tree）
        //         link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net+2);
        //     }
            

            
        //     //link_P_tree_to_P_pattern_location_dimension0.insert(link_P_tree_to_P_pattern_location_dimension0.end(),TorchEdges.size()-Torchedge.size(),TorchEdges.size());
        //     //link_P_tree_to_P_pattern_location_dimension1.insert(link_P_tree_to_P_pattern_location_dimension1.end(),Torchedge.size(),net_count*Max_tree_num_in_single_net+1);
        //     //std::cout<<"net_count"<<net_count<<std::endl;
        //     net_count++;
        //     //link_P_tree_to_P_pattern_location_dimension1末尾增加net_count个Torchedge.size()
        //     // TorchEdges[netIndex].resize(1);
        //     // TorchEdges[netIndex][0].insert(TorchEdges[netIndex][0].end(),Torchedge.begin(),Torchedge.end());//将上面的twopinnet加入到全部的矩阵序列里面去
        //     PatternRoutes_second.insert(std::make_pair(netIndex, patternRoute));
        // }
        // log()<<"generate_treeforest_finish"<<std::endl;
        // logeol();

        vector<int> alltwopinnet_num_within_batch(ceil(float(net_num)/batch_size));
        vector<int> net_num_within_batch(ceil(float(net_num)/batch_size),batch_size);
        if (net_num%batch_size!=0) net_num_within_batch[ceil(float(net_num)/batch_size)-1]=net_num%batch_size;
        for (int batch_index=0;batch_index<ceil(float(net_num)/batch_size);batch_index++){

        
            alltwopinnet_num_within_batch[batch_index]=TorchEdges_batch[batch_index].size();
            link_P_tree_to_P_pattern_location_dimension0_batch[batch_index].clear();
            link_P_tree_to_P_pattern_location_dimension0_batch[batch_index].resize(alltwopinnet_num_within_batch[batch_index]);
            // std::cout<<link_P_tree_to_P_pattern_location_dimension0_batch[batch_index].size()<<std::endl;
            // log() <<alltwopinnet_num_within_batch[batch_index]<<":batch["<<batch_index<<"]"<< std::endl;
            for (int i = 0; i < alltwopinnet_num_within_batch[batch_index]; i++) link_P_tree_to_P_pattern_location_dimension0_batch[batch_index][i]=i;//link_P_tree_to_P_pattern_location_dimension0.push_back(i);//
            // std::cout<<link_P_tree_to_P_pattern_location_dimension0_batch[batch_index].size()<<std::endl;
            // std::cout << "finish" << std::endl;
            //std::cout << link_P_tree_to_P_pattern_location_dimension0.size() << std::endl;
            //std::cout << link_P_tree_to_P_pattern_location_dimension1.size() << std::endl;

            torch::Tensor link_P_tree_to_P_pattern_location_dimension0_tensor=torch::tensor(link_P_tree_to_P_pattern_location_dimension0,torch::kLong);
            //std::cout << "1" << std::endl;
            torch::Tensor link_P_tree_to_P_pattern_location_dimension1_tensor=torch::tensor(link_P_tree_to_P_pattern_location_dimension1,torch::kLong);
            //std::cout << "1" << std::endl;
            torch::Tensor link_P_tree_to_P_pattern_location_tensor=torch::stack({link_P_tree_to_P_pattern_location_dimension0_tensor,link_P_tree_to_P_pattern_location_dimension1_tensor});
            //std::cout << "1" << std::endl;
            torch::Tensor link_P_tree_to_P_pattern_location_values_tensor=torch::ones(link_P_tree_to_P_pattern_location_dimension0_tensor.sizes());
            //std::cout << "1" << std::endl;

            link_P_tree_to_P_pattern_batch[batch_index] = torch::sparse_coo_tensor(link_P_tree_to_P_pattern_location_tensor,link_P_tree_to_P_pattern_location_values_tensor,{alltwopinnet_num_within_batch[batch_index],net_num_within_batch[batch_index]*Max_tree_num_in_single_net}).to(torch::kFloat).to(dev);
        }




        // int alltwopinnet_num=TorchEdges.size();
        // link_P_tree_to_P_pattern_location_dimension0.clear();
        // link_P_tree_to_P_pattern_location_dimension0.resize(alltwopinnet_num);
        // std::cout<<link_P_tree_to_P_pattern_location_dimension0.size()<<std::endl;
        // log() <<alltwopinnet_num<< std::endl;
        // for (int i = 0; i < alltwopinnet_num; i++) link_P_tree_to_P_pattern_location_dimension0[i]=i;//link_P_tree_to_P_pattern_location_dimension0.push_back(i);//
        // std::cout<<link_P_tree_to_P_pattern_location_dimension0.size()<<std::endl;
        // // std::cout << "finish" << std::endl;
        // //std::cout << link_P_tree_to_P_pattern_location_dimension0.size() << std::endl;
        // //std::cout << link_P_tree_to_P_pattern_location_dimension1.size() << std::endl;

        // torch::Tensor link_P_tree_to_P_pattern_location_dimension0_tensor=torch::tensor(link_P_tree_to_P_pattern_location_dimension0,torch::kLong);
        // //std::cout << "1" << std::endl;
        // torch::Tensor link_P_tree_to_P_pattern_location_dimension1_tensor=torch::tensor(link_P_tree_to_P_pattern_location_dimension1,torch::kLong);
        // //std::cout << "1" << std::endl;
        // torch::Tensor link_P_tree_to_P_pattern_location_tensor=torch::stack({link_P_tree_to_P_pattern_location_dimension0_tensor,link_P_tree_to_P_pattern_location_dimension1_tensor});
        // //std::cout << "1" << std::endl;
        // torch::Tensor link_P_tree_to_P_pattern_location_values_tensor=torch::ones(link_P_tree_to_P_pattern_location_dimension0_tensor.sizes());
        // //std::cout << "1" << std::endl;

        // link_P_tree_to_P_pattern=torch::sparse_coo_tensor(link_P_tree_to_P_pattern_location_tensor,link_P_tree_to_P_pattern_location_values_tensor,{alltwopinnet_num,net_num*Max_tree_num_in_single_net}).to(torch::kFloat).to(dev);
        
        
        // // link_P_tree_to_P_pattern_location_dimension0_tensor.reset();
        // // link_P_tree_to_P_pattern_location_dimension1_tensor.reset();
        // // link_P_tree_to_P_pattern_location_tensor.reset();
        // // link_P_tree_to_P_pattern_location_values_tensor.reset();

        // link_P_tree_to_P_pattern_location_dimension0_tensor.reset();
        // link_P_tree_to_P_pattern_location_dimension1_tensor.reset();
        // link_P_tree_to_P_pattern_location_tensor.reset();
        // link_P_tree_to_P_pattern_location_values_tensor.reset();






        //optimize here
        // c10::DeviceType dev;
        // get_optim_device(&dev);
        
        // link_P_tree_to_P_pattern=(link_P_tree_to_P_pattern).to(torch::kFloat).to(dev);
        
        log()<<"link_P_tree_to_P_pattern_mask_finish"<<std::endl;
        logeol();
        
        

        // //以下的生成2d的capacitymap的方法是直接基于三维的，但是这里有个问题是三维的vector不能直接转变成tensor
        // std::vector<::vector<std::vector<int>>> compresed_capacitymap(2,std::vector<std::vector<int>>(gcell_num_x,std::vector<int>(gcell_num_y)));//layer,x,y的顺序
        // for (int y = 0; y < gridGraph.getSize(1); y++)
        // {
        //     for (int x = 0; x < gridGraph.getSize(0); x++)
        //     {
        //         //for (int hlayer=0 ; hlayer<gridGraph.getNumLayers(); hlayer+=2) compresed_capacitymap[0][x][y]+=GlobalRouter::capacitymap[hlayer][y][x];
        //         for (int hlayer=0 ; hlayer<gridGraph.getNumLayers(); hlayer+=2) compresed_capacitymap[0][x][y]+= gridGraph.getEdge(hlayer,x,y).capacity;
        //         //for (int vlayer=1 ; vlayer<gridGraph.getNumLayers(); vlayer+=2) compresed_capacitymap[1][x][y]+=GlobalRouter::capacitymap[vlayer][y][x];
        //         for (int vlayer=1 ; vlayer<gridGraph.getNumLayers(); vlayer+=2) compresed_capacitymap[1][x][y]+= gridGraph.getEdge(vlayer,x,y).capacity;
        //     }
            
        // }
        // torch::Tensor compresed_capacity_map_tensor = torch::zeros({2, gridGraph.getSize(0), gridGraph.getSize(1)}, torch::kFloat);
        // for (int i = 0; i < 2; ++i) {
        //     for (int x = 0; x < gridGraph.getSize(0); ++x) {
        //         for (int y = 0; y < gridGraph.getSize(1); ++y) {
        //             compresed_capacity_map_tensor[i][x][y] = compresed_capacitymap[i][x][y];
        //         }
        //     }
        // }

        

        // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
        // int net_num=nets.size();
        // int Max_tree_num_in_single_net=0;
        // int Max_twopinnet_num_in_single_tree=0;
        // for (const auto& net : TorchEdges){
        //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
        //     for (const auto& tree : net){
        //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
        //     }
        // }


        //以下一段可以获取alltwopinnet_num
        
        //int Max_tree_num_in_single_net=0;
        //int Max_twopinnet_num_in_single_tree=0;
        
        


        // //以下一段代码生成针对P矩阵中的所有变量进行有效位和无效位的标记，P矩阵的有效位置为1，无效位置为-inf（或者-1e9）
        // torch::Tensor P_pattern_array_invalid_equals_to_negtive_infinity_mask=torch::full({net_num,Max_tree_num_in_single_net,Max_twopinnet_num_in_single_tree,2},1,dev);
        // //torch::Tensor P_pattern_array_invalid_equals_to_negtive_infinity_mask=torch::full({net_num,Max_tree_num_in_single_net,Max_twopinnet_num_in_single_tree,},-std::numeric_limits<int>::infinity(),dev);
        // for (int net_index=0;net_index<TorchEdges.size();net_index++){
        //     for(int tree_index=0;tree_index<TorchEdges[net_index].size();tree_index++){
        //         for(int twopinnet_index=0;twopinnet_index<TorchEdges[net_index][tree_index].size();twopinnet_index++){
        //             P_pattern_array_invalid_equals_to_negtive_infinity_mask[net_index][tree_index][twopinnet_index][0]=0;
        //             P_pattern_array_invalid_equals_to_negtive_infinity_mask[net_index][tree_index][twopinnet_index][1]=0;
        //         }
        //     }
        // }
        





        //以下是分批优化版本
        vector<torch::Tensor> demand_mask_batch(ceil(float(net_num)/batch_size));
        vector<torch::Tensor> viacount_mask_batch(ceil(float(net_num)/batch_size));
        
        std::vector<int> Pattern_1d_vector;
        std::vector<int> tree_selection_1d_vector;
        std::vector<int> tree_next_update_1d_vector;
        vector<double> time_opt(5,0);
        for(int batch_index=0;batch_index<ceil(float(net_num)/batch_size);batch_index++){
            // create_masks_fixed(2,gcell_num_x,gcell_num_y,TorchEdges_batch,&demand_mask,dev);
            auto t0=std::chrono::high_resolution_clock::now();
            
            create_masks_fixed_agian_batch(2,gcell_num_x,gcell_num_y,TorchEdges_batch[batch_index],&demand_mask_batch[batch_index],&wirelength_mask,&viacount_mask_batch[batch_index],dev);
            log()<<"generate_net_pass_through_gcell_mask_finish"<<std::endl;
            logeol();
            
            auto t1=std::chrono::high_resolution_clock::now();
            
            auto route = Torchroute(net_num_within_batch[batch_index],Max_tree_num_in_single_net,alltwopinnet_num_within_batch[batch_index],2);//int net_num,int max_tree_num_in_one_single_net, int two_pin_net_num, int pattern_num
            route.train();
            route.to(dev);
            torch::optim::Adam optimizer(route.parameters(), torch::optim::AdamOptions(0.3));
        
            auto t2=std::chrono::high_resolution_clock::now();

            torch::Tensor this_batch_all_gcell_demand;
            float lastloss = 0.0;
            int epoch = 0;
            int max_epochs_num = 100;
            for (epoch = 0; epoch < max_epochs_num; epoch++)
            {
                optimizer.zero_grad();
                //torch::Tensor loss = route.forward(TorchEdges.size(), 2, mask_h, mask_v, capacity, gcell_num_x, gcell_num_y, dev);
                // torch::Tensor loss = route.forwardfixed_agian(net_num,Max_tree_num_in_single_net,TorchEdges.size(), 2, &demand_mask,&wirelength_mask,&viacount_mask,&link_P_tree_to_P_pattern, compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev);
                //torch::Tensor loss = route.forwardfixed(net_num,Max_tree_num_in_single_net,TorchEdges.size(), 2, &demand_mask,&wirelength_mask,&viacount_mask,&link_P_tree_to_P_pattern, compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev);
                torch::Tensor loss = route.forwardfixed_batch(net_num_within_batch[batch_index],Max_tree_num_in_single_net,TorchEdges_batch[batch_index].size(), 2,&demand_mask_batch[batch_index],&wirelength_mask,&viacount_mask_batch[batch_index],&link_P_tree_to_P_pattern_batch[batch_index], compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev,batch_index,batch_size,this_batch_all_gcell_demand);
                //torch::Tensor loss = route.forwardfixed_batch(net_num,Max_tree_num_in_single_net,TorchEdges.size(), 2,&demand_mask_batch[batch_index],&wirelength_mask,&viacount_mask,&link_P_tree_to_P_pattern, compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev,batch_index,batch_size);
                loss.backward();
                optimizer.step();
                // if (epoch==max_epochs_num-1) {compresed_capacity_map_tensor=torch::sub(compresed_capacity_map_tensor,this_batch_all_gcell_demand).to(dev);}
                // cout << route.parameters() << endl;

                // if (abs(loss.item().to<float>() - lastloss) / lastloss < 0.0002)
                // {
                //     lastloss = loss.item().to<float>();
                //     break;
                // }
                // else
                // {
                //     lastloss = loss.item().to<float>();
                //     std::cout << "loss: " << loss << std::endl;
                // }
            }
            auto t3=std::chrono::high_resolution_clock::now();

            compresed_capacity_map_tensor=torch::sub(compresed_capacity_map_tensor,this_batch_all_gcell_demand.detach()).to(dev);
            log() << "@epoch: " << epoch << ", Torch optimization stopped." << std::endl;
            logeol();
        

            std::cout << route.parameters().size() << std::endl;
            std::cout << route.parameters()[0].sizes() << std::endl;
            std::cout << route.parameters()[1].sizes() << std::endl;
            
            //auto Pattern = torch::argmax(route.parameters()[0], 3).to(dev);
            auto Pattern = torch::argmax(route.parameters()[0], 1).to(torch::kCPU).contiguous();
            auto Tree_selection = torch::argmax(route.parameters()[1], 1).to(torch::kCPU).contiguous();
            auto Tree_next_update = torch::argmin(route.parameters()[1], 1).to(torch::kCPU).contiguous();
            std::cout << Pattern.dtype()<< std::endl;
            std::cout << Tree_selection.dtype()<< std::endl;

            auto t4=std::chrono::high_resolution_clock::now();


            long* Pattern_ptr = Pattern.data_ptr<long>();
            long* Tree_selection_ptr = Tree_selection.data_ptr<long>();
            long* Tree_next_update_ptr = Tree_next_update.data_ptr<long>();
            //std::cout<<"ok"<<std::endl;
            std::vector<int> Pattern_1d_vector_temp(Pattern_ptr, Pattern_ptr + Pattern.numel());
            std::vector<int> tree_selection_1d_vector_temp(Tree_selection_ptr, Tree_selection_ptr + Tree_selection.numel());
            std::vector<int> tree_next_update_1d_vector_temp(Tree_next_update_ptr, Tree_next_update_ptr + Tree_next_update.numel());

            auto t5=std::chrono::high_resolution_clock::now();

            Pattern_1d_vector.insert(Pattern_1d_vector.end(),Pattern_1d_vector_temp.begin(),Pattern_1d_vector_temp.end());
            tree_selection_1d_vector.insert(tree_selection_1d_vector.end(),tree_selection_1d_vector_temp.begin(),tree_selection_1d_vector_temp.end());
            tree_next_update_1d_vector.insert(tree_next_update_1d_vector.end(),tree_next_update_1d_vector_temp.begin(),tree_next_update_1d_vector_temp.end());

            time_opt[0]+=std::chrono::duration<double>(t1-t0).count();
            time_opt[1]+=std::chrono::duration<double>(t2-t1).count();
            time_opt[2]+=std::chrono::duration<double>(t3-t2).count();
            time_opt[3]+=std::chrono::duration<double>(t4-t3).count();
            time_opt[4]+=std::chrono::duration<double>(t5-t4).count();
            // time_opt[5]+=std::chrono::duration<double>(t2-t1).count();

        }
        std::cout<<time_opt[0]<<" "<<time_opt[1]<<" "<<time_opt[2]<<" "<<time_opt[3]<<" "<<time_opt[4]<<" "<<std::endl;
        tree_selection_vector = tree_selection_1d_vector;
        tree_next_update_vector = tree_next_update_1d_vector;
        






        // //以下是所有一块优化版本
        // // create_masks_fixed(2,gcell_num_x,gcell_num_y,TorchEdges,&demand_mask,dev);
        // create_masks_fixed_agian(2,gcell_num_x,gcell_num_y,TorchEdges,&demand_mask,&wirelength_mask,&viacount_mask,dev);
        // log()<<"generate_net_pass_through_gcell_mask_finish"<<std::endl;
        // logeol();
        // auto route = Torchroute(net_num,Max_tree_num_in_single_net,alltwopinnet_num,2);//int net_num,int max_tree_num_in_one_single_net, int two_pin_net_num, int pattern_num
        // route.train();
        // route.to(dev);
        // torch::optim::Adam optimizer(route.parameters(), torch::optim::AdamOptions(0.3));
        // float lastloss = 0.0;
        // int epoch = 0;
        // int max_epochs_num = 100;
        // for (epoch = 0; epoch < max_epochs_num; epoch++)
        // {
        //     optimizer.zero_grad();
            
        //     //torch::Tensor loss = route.forward(TorchEdges.size(), 2, mask_h, mask_v, capacity, gcell_num_x, gcell_num_y, dev);
        //     // torch::Tensor loss = route.forwardfixed_agian(net_num,Max_tree_num_in_single_net,TorchEdges.size(), 2, &demand_mask,&wirelength_mask,&viacount_mask,&link_P_tree_to_P_pattern, compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev);
        //     torch::Tensor loss = route.forwardfixed(net_num,Max_tree_num_in_single_net,TorchEdges.size(), 2, &demand_mask,&wirelength_mask,&viacount_mask,&link_P_tree_to_P_pattern, compresed_capacity_map_tensor,2,gcell_num_x,gcell_num_y,dev);
        //     loss.backward();
        //     optimizer.step();
        //     // cout << route.parameters() << endl;

        //     // if (abs(loss.item().to<float>() - lastloss) / lastloss < 0.0002)
        //     // {
        //     //     lastloss = loss.item().to<float>();
        //     //     break;
        //     // }
        //     // else
        //     // {
        //     //     lastloss = loss.item().to<float>();
        //     //     std::cout << "loss: " << loss << std::endl;
        //     // }
        // }
        // log() << "@epoch: " << epoch << ", Torch optimization stopped." << std::endl;
        // logeol();
        // std::cout << route.parameters().size() << std::endl;
        // std::cout << route.parameters()[0].sizes() << std::endl;
        // std::cout << route.parameters()[1].sizes() << std::endl;
        
        // //auto Pattern = torch::argmax(route.parameters()[0], 3).to(dev);
        // auto Pattern = torch::argmax(route.parameters()[0], 1).to(torch::kCPU).contiguous();
        // auto Tree_selection = torch::argmax(route.parameters()[1], 1).to(torch::kCPU).contiguous();
        // auto Tree_next_update = torch::argmin(route.parameters()[1], 1).to(torch::kCPU).contiguous();
        // std::cout << Pattern.dtype()<< std::endl;
        // std::cout << Tree_selection.dtype()<< std::endl;


        // long* Pattern_ptr = Pattern.data_ptr<long>();
        // long* Tree_selection_ptr = Tree_selection.data_ptr<long>();
        // long* Tree_next_update_ptr = Tree_next_update.data_ptr<long>();
        // //std::cout<<"ok"<<std::endl;
        // std::vector<int> Pattern_1d_vector(Pattern_ptr, Pattern_ptr + Pattern.numel());
        // std::vector<int> tree_selection_1d_vector(Tree_selection_ptr, Tree_selection_ptr + Tree_selection.numel());
        // std::vector<int> tree_next_update_1d_vector(Tree_next_update_ptr, Tree_next_update_ptr + Tree_next_update.numel());
        // tree_selection_vector = tree_selection_1d_vector;
        // tree_next_update_vector = tree_next_update_1d_vector;
















        // std::cout<<Pattern_1d_vector.size()<<std::endl;
        // std::cout<<Pattern_1d_vector[0]<<std::endl;
        // std::cout<<Pattern_1d_vector[1]<<std::endl;
        // std::cout<<Pattern_1d_vector[2]<<std::endl;
        // std::cout<<Pattern_1d_vector[3]<<std::endl;
        // std::cout<<Pattern_1d_vector[4]<<std::endl;
        // std::cout<<Pattern_1d_vector[5]<<std::endl;
        // std::cout<<Pattern_1d_vector[6]<<std::endl;
        // std::cout<<tree_selection_1d_vector.size()<<std::endl;
        // std::cout<<tree_next_update_1d_vector.size()<<std::endl;
        // std::cout<<"ok"<<std::endl;
        // std::cout<<tree_selection_vector[0]<<std::endl;
        // std::cout<<tree_selection_vector[1]<<std::endl;
        // std::cout<<tree_selection_vector[2]<<std::endl;
        // std::cout<<tree_selection_vector[3]<<std::endl;
        // std::cout<<tree_selection_vector[4]<<std::endl;
        // std::cout<<tree_selection_vector[5]<<std::endl;
        // std::cout<<tree_selection_vector[6]<<std::endl;
        // std::cout<<"ok"<<std::endl;
        size_t twopinnet_count = 0;
        for(int net_index=0;net_index<net_num;net_index++){
            for(int tree_index=0;tree_index<Max_tree_num_in_single_net;tree_index++){
                for(int twopinnet_index=0;twopinnet_index<TorchEdges0[net_index][tree_index].size();twopinnet_index++){
                    pattern_selection[net_index][tree_index].push_back(Pattern_1d_vector[twopinnet_count]);//这里能push_back吗
                    twopinnet_count++;
                }
                //std::cout<<"TorchEdges0["<<net_index<<"]["<<tree_index<<"].size()"<<TorchEdges0[net_index][tree_index].size()<<std::endl;
                //std::cout<<"pattern_selection["<<net_index<<"]["<<tree_index<<"]"<<pattern_selection[net_index][tree_index]<<std::endl;

            }

        }
        // std::cout<<"Pattern_1d_vector"<<Pattern_1d_vector<<std::endl;
        // std::cout<<Pattern_1d_vector[0]<<std::endl;
        // std::cout<<Pattern_1d_vector[1]<<std::endl;
        // std::cout<<Pattern_1d_vector[2]<<std::endl;
        // std::cout<<Pattern_1d_vector[3]<<std::endl;
        // std::cout<<Pattern_1d_vector[4]<<std::endl;
        // std::cout<<Pattern_1d_vector[5]<<std::endl;
        // std::cout<<Pattern_1d_vector[6]<<std::endl;

        // std::cout<<"pattern_selection"<<pattern_selection<<std::endl;
        // std::cout<<"pattern_selection[0][0]"<<pattern_selection[0][0]<<std::endl;
        // std::cout<<"pattern_selection[0][1]"<<pattern_selection[0][1]<<std::endl;
        // std::cout<<"pattern_selection[1][0]"<<pattern_selection[1][0]<<std::endl;
        // std::cout<<"pattern_selection[1][1]"<<pattern_selection[1][1]<<std::endl;
        // std::cout<<"pattern_selection[2][0]"<<pattern_selection[2][0]<<std::endl;
        // std::cout<<"pattern_selection[2][1]"<<pattern_selection[2][1]<<std::endl;
        // std::cout<<"pattern_selection[3][0]"<<pattern_selection[3][0]<<std::endl;
        // std::cout<<"pattern_selection[3][1]"<<pattern_selection[3][1]<<std::endl;


        // //根据Torchedges0的形状复原pattern和treeselection
        // int twopinnet_count=0;
        // for(int net_index=0;net_index<net_num;net_index++){
        //     for(int tree_index=0;tree_index<Max_tree_num_in_single_net;tree_index++){
                
        //         for(int twopinnet_index=0;twopinnet_index<TorchEdges0[net_index][tree_index].size();twopinnet_index++){
        //             //pattern_selection[net_index][tree_index].emplace_back(Pattern[twopinnet_count].item().to<int>());
        //             pattern_selection[net_index][tree_index].push_back(Pattern[twopinnet_count].item().to<int>());
                    
        //             twopinnet_count++;
        //         }
                
        //         pattern_selection[net_index][tree_index].insert(pattern_selection[net_index][tree_index].end(),10,1);
        //         //log()<<pattern_selection[net_index][tree_index].size()<<std::endl;
        //     }
        //     tree_selection_vector[net_index]=Tree_selection[net_index].item().to<int>();
        //     tree_next_update_vector[net_index]=Tree_next_update[net_index].item().to<int>();
        // }

        // //根据Torchedges0的形状复原pattern和treeselection
        // vector<vector<vector<int>>> pattern_selection(net_num,vector<vector<int>>(Max_tree_num_in_single_net));
        // vector<int> tree_selection_vector(net_num);
        // int twopinnet_count=0;
        // //long int* pattern_selection_ptr = Pattern.data<long int>();
        // //long int* tree_selection_ptr = Tree_selection.data<long int>();
        // std::cout<<"ok"<<std::endl;
        // // std::vector<long int> pattern_selection_vector_1d(pattern_selection_ptr,pattern_selection_ptr+Pattern.numel());
        // // std::vector<long int> tree_selection_vector_1d(tree_selection_ptr,tree_selection_ptr+Tree_selection.numel());

        // std::vector<int> pattern_selection_vector_1d(Pattern.data_ptr<int64_t>(),Pattern.data_ptr<int64_t>()+Pattern.numel());
        // std::vector<int> tree_selection_vector_1d(Tree_selection.data_ptr<int64_t>(),Tree_selection.data_ptr<int64_t>()+Tree_selection.numel());
        
        // for(int net_index=0;net_index<net_num;net_index++){
        //     for(int tree_index=0;tree_index<Max_tree_num_in_single_net;tree_index++){
                
        //         for(int twopinnet_index=0;twopinnet_index<TorchEdges0[net_index][tree_index].size();twopinnet_index++){
        //             //pattern_selection[net_index][tree_index].emplace_back(Pattern[twopinnet_count].item().to<int>());
        //             //pattern_selection[net_index][tree_index].push_back(Pattern[twopinnet_count].item().to<int>());
        //             pattern_selection[net_index][tree_index].push_back(pattern_selection_vector_1d[twopinnet_count]);
                    
                    
        //             twopinnet_count++;
        //         }
                
        //         //pattern_selection[net_index][tree_index].insert(pattern_selection[net_index][tree_index].end(),10,1);
        //         //log()<<pattern_selection[net_index][tree_index].size()<<std::endl;
        //     }
        //     tree_selection_vector[net_index]=tree_selection_vector_1d[net_index];
        // }
        
        //auto tree = torch::argmax(route.parameters()[1],1);
        // std::cout << Pattern.sizes() << std::endl;


        netIndices.clear();
        netIndices.reserve(nets.size());
        for (const auto& net : nets) netIndices.push_back(net.getIndex());
        sortNetIndices(netIndices);



        log() << "layer assignment and Commit Trees..." << std::endl;
        //int two_pin_net_index = 0;
        
        assert(tree_selection_vector.size()==netIndices.size());
        assert(Pattern_1d_vector.size()==TorchEdges.size());
        for (const int netIndex : netIndices)
        {
            // auto patternRoute = PatternRoutes.find(netIndex)->second;
            // //if (tree_selection_vector[netIndex]&&enter_pretend_stage2_flag[netIndex]) patternRoute.constructRoutingDAG_based_on_routingTree(nets[netIndex].getRoutingTree());//若为1，则说明选了第二棵树
            // if (enter_pretend_stage2_flag[netIndex][tree_selection_vector[netIndex]]) patternRoute.constructRoutingDAG_based_on_routingTree(nets[netIndex].getRoutingTree());//若为1，则说明选了第二棵树
            // else patternRoute.constructRoutingDAGfixed55(pattern_selection[netIndex][tree_selection_vector[netIndex]]);

            nets[netIndex].clearRoutingTree();
            //auto patternRoute = PatternRoutes.find(netIndex)->second;
            //auto patternRoute = tree_selection_vector[netIndex]?PatternRoutes_second.find(netIndex)->second:PatternRoutes.find(netIndex)->second;
            // auto patternRoute = tree_selection_vector[netIndex]?PatternRoutes_multi[1].find(netIndex)->second:PatternRoutes_multi[0].find(netIndex)->second;
            auto patternRoute = PatternRoutes_multi[tree_selection_vector[netIndex]].find(netIndex)->second;
            //std::cout<<tree_selection_vector[netIndex]<<" ";
            // std::cout<<"ok"<<std::endl;
            //std::cout<<tree_selection_vector[netIndex]<<std::endl;
            //std::cout<<pattern_selection[netIndex][tree_selection_vector[netIndex]]<<std::endl;
            // std::cout<<netIndex<<std::endl;
            patternRoute.constructRoutingDAGfixed55(pattern_selection[netIndex][tree_selection_vector[netIndex]]);
            // PatternRoutes_stage2.insert(std::make_pair(netIndex, patternRoute));

            //std::cout<<std::endl;
            // PatternRoute patternRoute(nets[netIndex], gridGraph, parameters);
            // patternRoute.constructSteinerTree();
            //std::cout << "netIndex: " << netIndex << std::endl;
            //patternRoute.constructRoutingDAG();
            //patternRoute.constructRoutingDAGfixed(Pattern,two_pin_net_index);
            //patternRoute.constructRoutingDAGfixed55(pattern_selection[netIndex][tree_selection_vector[netIndex]]);
            //patternRoute.constructRoutingDAGfixed_again(Pattern,netIndex);
            //std::cout << "constructRoutingDAGfixed_again" << std::endl;
            //PatternRoutes_stage2.insert(std::make_pair(netIndex, patternRoute));
            patternRoute.run();
            //std::cout << "run_finished" << std::endl;
            gridGraph.commitTree(nets[netIndex].getRoutingTree());
            //std::cout<<netIndex<<std::endl;
            
            
        }
        log()<<"finish layer assignment and Commit Trees..." << std::endl;
        logeol();
    }


    netIndices.clear();
    for (const auto& net : nets) {
        if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
            netIndices.push_back(net.getIndex());
        }
    }
    log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
    logeol();
    printStatistics();
    t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    
    

    // Stage 2: Pattern routing with possible detours
    n2 = netIndices.size();
    if (netIndices.size() > 0) {
        log() << "stage 2: pattern routing with possible detours" << std::endl;
        GridGraphView<bool> congestionView; // (2d) direction -> x -> y -> has overflow?
        gridGraph.extractCongestionView(congestionView);
        // for (const int netIndex : netIndices) {
        //     GRNet& net = nets[netIndex];
        //     gridGraph.commitTree(net.getRoutingTree(), true);
        // }
        sortNetIndices(netIndices);
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            
            gridGraph.commitTree(net.getRoutingTree(), true);
            net.clearRoutingTree();


            // PatternRoute patternRoute(net, gridGraph, parameters);
            // patternRoute.constructSteinerTree();
            // patternRoute.constructRoutingDAG();


            auto patternRoute = PatternRoutes_multi[tree_selection_vector[netIndex]].find(netIndex)->second;
            //patternRoute.constructRoutingDAGfixed55(pattern_selection[netIndex][tree_selection_vector[netIndex]]);
            patternRoute.constructRoutingDAG();


            

            //auto patternRoute = PatternRoutes_stage2.find(netIndex)->second;
            
            
            // std::cout<<"2"<<std::endl;
            // std::cout << "choose"<<enter_pretend_stage2_flag[netIndex][tree_selection_vector[netIndex]]<<std::endl;
            // auto patternRoute = PatternRoutes_stage2.find(netIndex)->second;
            // std::cout<<"3"<<std::endl;
            //auto patternRoute = tree_selection_vector[netIndex]?PatternRoutes_second.find(netIndex)->second:PatternRoutes.find(netIndex)->second;


            patternRoute.constructDetours(congestionView); // KEY DIFFERENCE compared to stage 1
            
            patternRoute.run();
            gridGraph.commitTree(net.getRoutingTree());
        }
        
        netIndices.clear();
        for (const auto& net : nets) {
            if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
                netIndices.push_back(net.getIndex());
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        logeol();
        
    }
    
    t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    printStatistics();
    // Stage 3: maze routing on sparsified routing graph
    n3 = netIndices.size();
    if (netIndices.size() > 0) {
        log() << "stage 3: maze routing on sparsified routing graph" << std::endl;
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            gridGraph.commitTree(net.getRoutingTree(), true);
        }
        GridGraphView<CostT> wireCostView;
        gridGraph.extractWireCostView(wireCostView);
        sortNetIndices(netIndices);
        SparseGrid grid(10, 10, 0, 0);
        for (const int netIndex : netIndices) {
            GRNet& net = nets[netIndex];
            // gridGraph.commitTree(net.getRoutingTree(), true);
            // gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
            MazeRoute mazeRoute(net, gridGraph, parameters);
            mazeRoute.constructSparsifiedGraph(wireCostView, grid);
            mazeRoute.run();
            std::shared_ptr<SteinerTreeNode> tree = mazeRoute.getSteinerTree();
            assert(tree != nullptr);
            
            PatternRoute patternRoute(net, gridGraph, parameters);
            patternRoute.setSteinerTree(tree);
            patternRoute.constructRoutingDAG();
            patternRoute.run();
            
            gridGraph.commitTree(net.getRoutingTree());
            gridGraph.updateWireCostView(wireCostView, net.getRoutingTree());
            grid.step();
        }
        netIndices.clear();
        for (const auto& net : nets) {
            if (gridGraph.checkOverflow(net.getRoutingTree()) > 0) {
                netIndices.push_back(net.getIndex());
            }
        }
        log() << netIndices.size() << " / " << nets.size() << " nets have overflows." << std::endl;
        log() << "finish stage 3" << std::endl;
        logeol();
    }
    
    t3 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count();
    t = std::chrono::high_resolution_clock::now();
    
    // std::cout << "iteration statistics " 
    //     << n1 << " " << std::setprecision(3) << std::fixed << t1 << " " 
    //     << n2 << " " << std::setprecision(3) << std::fixed << t2 << " " 
    //     << n3 << " " << std::setprecision(3) << std::fixed << t3 << std::endl;
    
    printStatistics();
    if (parameters.write_heatmap) gridGraph.write();
}

void GlobalRouter::sortNetIndices(vector<int>& netIndices) const {
    vector<int> halfParameters(nets.size());
    for (int netIndex : netIndices) {
        auto& net = nets[netIndex];
        halfParameters[netIndex] = net.getBoundingBox().hp();
    }
    sort(netIndices.begin(), netIndices.end(), [&](int lhs, int rhs) {
        return halfParameters[lhs] < halfParameters[rhs];
    });
}

void GlobalRouter::getGuides(const GRNet& net, vector<std::pair<int, utils::BoxT<int>>>& guides) {
    auto& routingTree = net.getRoutingTree();
    if (!routingTree) return;
    // 0. Basic guides
    GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
        for (const auto& child : node->children) {
            if (node->layerIdx == child->layerIdx) {
                guides.emplace_back(
                    node->layerIdx, utils::BoxT<int>(
                        min(node->x, child->x), min(node->y, child->y),
                        max(node->x, child->x), max(node->y, child->y)
                    )
                );
            } else {
                int maxLayerIndex = max(node->layerIdx, child->layerIdx);
                for (int layerIdx = min(node->layerIdx, child->layerIdx); layerIdx <= maxLayerIndex; layerIdx++) {
                    guides.emplace_back(layerIdx, utils::BoxT<int>(node->x, node->y));
                }
            }
        }
    });
    
    
    auto getSpareResource = [&] (const GRPoint& point) {
        double resource = std::numeric_limits<double>::max();
        unsigned direction = gridGraph.getLayerDirection(point.layerIdx);
        if (point[direction] + 1 < gridGraph.getSize(direction)) {
            resource = min(resource, gridGraph.getEdge(point.layerIdx, point.x, point.y).getResource());
        }
        if (point[direction] > 0) {
            GRPoint lower = point;
            lower[direction] -= 1;
            resource = min(resource, gridGraph.getEdge(lower.layerIdx, point.x, point.y).getResource());
        }
        return resource;
    };
    
    // 1. Pin access patches
    assert(parameters.min_routing_layer + 1 < gridGraph.getNumLayers());
    for (auto& gpts : net.getPinAccessPoints()) {
        for (auto& gpt : gpts) {
            if (gpt.layerIdx < parameters.min_routing_layer) {
                int padding = 0;
                if (getSpareResource({parameters.min_routing_layer, gpt.x, gpt.y}) < parameters.pin_patch_threshold) {
                    padding = parameters.pin_patch_padding;
                }
                for (int layerIdx = gpt.layerIdx; layerIdx <= parameters.min_routing_layer + 1; layerIdx++) {
                    guides.emplace_back(layerIdx, utils::BoxT<int>(
                        max(gpt.x - padding, 0),
                        max(gpt.y - padding, 0),
                        min(gpt.x + padding, (int)gridGraph.getSize(0) - 1),
                        min(gpt.y + padding, (int)gridGraph.getSize(1) - 1)
                    ));
                    areaOfPinPatches += (guides.back().second.x.range() + 1) * (guides.back().second.y.range() + 1);
                }
            }
        }
    }
    
    // 2. Wire segment patches
    GRTreeNode::preorder(routingTree, [&](std::shared_ptr<GRTreeNode> node) {
        for (const auto& child : node->children) {
            if (node->layerIdx == child->layerIdx) {
                double wire_patch_threshold = parameters.wire_patch_threshold;
                unsigned direction = gridGraph.getLayerDirection(node->layerIdx);
                int l = min((*node)[direction], (*child)[direction]);
                int h = max((*node)[direction], (*child)[direction]);
                int r = (*node)[1 - direction];
                for (int c = l; c <= h; c++) {
                    bool patched = false;
                    GRPoint point = (direction == MetalLayer::H ? GRPoint(node->layerIdx, c, r) : GRPoint(node->layerIdx, r, c));
                    if (getSpareResource(point) < wire_patch_threshold) {
                        for (int layerIndex = node->layerIdx - 1; layerIndex <= node->layerIdx + 1; layerIndex += 2) {
                            if (layerIndex < parameters.min_routing_layer || layerIndex >= gridGraph.getNumLayers()) continue;
                            if (getSpareResource({layerIndex, point.x, point.y}) >= 1.0) {
                                guides.emplace_back(layerIndex, utils::BoxT<int>(point.x, point.y));
                                areaOfWirePatches += 1;
                                patched = true;
                            }
                        }
                    } 
                    if (patched) {
                        wire_patch_threshold = parameters.wire_patch_threshold;
                    } else {
                        wire_patch_threshold *= parameters.wire_patch_inflation_rate;
                    }
                }
            }
        }
    });
}

void GlobalRouter::printStatistics() const {
    log() << "routing statistics" << std::endl;
    loghline();

    // wire length and via count
    uint64_t wireLength = 0;
    int viaCount = 0;
    vector<vector<vector<int>>> wireUsage;
    wireUsage.assign(
        gridGraph.getNumLayers(), vector<vector<int>>(gridGraph.getSize(0), vector<int>(gridGraph.getSize(1), 0))
    );
    for (const auto& net : nets) {
        GRTreeNode::preorder(net.getRoutingTree(), [&] (std::shared_ptr<GRTreeNode> node) {
            for (const auto& child : node->children) {
                if (node->layerIdx == child->layerIdx) {//如果是同层的（而且是处于同一条直线上的）
                    unsigned direction = gridGraph.getLayerDirection(node->layerIdx);
                    int l = min((*node)[direction], (*child)[direction]);//布线方向的小座标
                    int h = max((*node)[direction], (*child)[direction]);//布线方向的大坐标
                    int r = (*node)[1 - direction];//两个node的另一个坐标，即node和child处于同一条直线的直线坐标
                    for (int c = l; c < h; c++) {
                        wireLength += gridGraph.getEdgeLength(direction, c);
                        int x = direction == MetalLayer::H ? c : r;//如果是横向，x=c；如果是纵向，x=r
                        int y = direction == MetalLayer::H ? r : c;//如果是横向，y=r；如果是纵向，y=c
                        wireUsage[node->layerIdx][x][y] += 1;
                    }
                } else {
                    viaCount += abs(node->layerIdx - child->layerIdx);
                }
            }
        });
    }
    
    // resource
    CapacityT overflow = 0;

    CapacityT minResource = std::numeric_limits<CapacityT>::max();
    GRPoint bottleneck(-1, -1, -1);
    for (int layerIndex = parameters.min_routing_layer; layerIndex < gridGraph.getNumLayers(); layerIndex++) {
        unsigned direction = gridGraph.getLayerDirection(layerIndex);
        for (int x = 0; x < gridGraph.getSize(0) - 1 + direction; x++) {
            for (int y = 0; y < gridGraph.getSize(1) - direction; y++) {
                CapacityT resource = gridGraph.getEdge(layerIndex, x, y).getResource();
                if (resource < minResource) {
                    minResource = resource;
                    bottleneck = {layerIndex, x, y};
                }
                CapacityT usage = wireUsage[layerIndex][x][y];
                CapacityT capacity = max(gridGraph.getEdge(layerIndex, x, y).capacity, 0.0);
                if (usage > 0.0 && usage > capacity) {
                    overflow += usage - capacity;
                }
            }
        }
    }
    
    log() << "wire length (metric):  " << wireLength / gridGraph.getM2Pitch() << std::endl;
    log() << "total via count:       " << viaCount << std::endl;
    log() << "total wire overflow:   " << (int)overflow << std::endl;
    log() << "loss(metric)" << wireLength/ gridGraph.getM2Pitch()*0.5+viaCount*4+500*overflow <<std::endl;
    log() << "loss" << wireLength*0.5+viaCount*4+500*overflow <<std::endl;
    logeol();

    log() << "min resource: " << minResource << std::endl;
    log() << "bottleneck:   " << bottleneck << std::endl;

    logeol();
}

void GlobalRouter::write(std::string guide_file) {
    log() << "generating route guides..." << std::endl;
    if (guide_file == "") guide_file = parameters.out_file;
    
    areaOfPinPatches = 0;
    areaOfWirePatches = 0;
    std::stringstream ss;
    for (const GRNet& net : nets) {
        vector<std::pair<int, utils::BoxT<int>>> guides;
        getGuides(net, guides);
        
        ss << net.getName() << std::endl;
        ss << "(" << std::endl;
        for (const auto& guide : guides) {
            ss << gridGraph.getGridline(0, guide.second.x.low) << " "
                 << gridGraph.getGridline(1, guide.second.y.low) << " "
                 << gridGraph.getGridline(0, guide.second.x.high + 1) << " "
                 << gridGraph.getGridline(1, guide.second.y.high + 1) << " "
                 << gridGraph.getLayerName(guide.first) << std::endl;
        }
        ss << ")" << std::endl;
    }
    log() << "total area of pin access patches: " << areaOfPinPatches << std::endl;
    log() << "total area of wire segment patches: " << areaOfWirePatches << std::endl;
    log() << std::endl;
    log() << "writing output..." << std::endl;
    std::ofstream fout(guide_file);
    fout << ss.str();
    fout.close();
    log() << "finished writing output..." << std::endl;
}


//Modified by IrisLin
c10::DeviceType* GlobalRouter::get_optim_device(c10::DeviceType *dev)
{
	if (torch::cuda::is_available())
	{
        std::cout<<torch::cuda::device_count()<<std::endl;
        
		
        //c10::cuda::CUDAGuard::set_device(1);
        //at::cuda::_set_device(1);
        //torch::cuda::set_device(0);
		//*dev = at::kCUDA;
        *dev = torch::kCUDA;
        torch::Device device(torch::kCUDA, 1);
        //std::cout<<device<<std::end;
        *dev = device.type();
	}
	else
	{
		//*dev = at::kCPU;
        *dev = torch::kCPU;
	}
	// Change device to CPU by uncommenting the following line:
	// *dev = at::kCPU;
	log() << "current optimization device: " << *dev << std::endl;
    logeol();
	return dev;
}
// c10::Device* GlobalRouter::get_optim_device_fix(c10::Device dev)
// {
// 	if (torch::cuda::is_available())
// 	{
//         std::cout<<torch::cuda::device_count()<<std::endl;
        
		
//         //c10::cuda::CUDAGuard::set_device(1);
//         //at::cuda::_set_device(1);
//         //torch::cuda::set_device(0);
// 		//*dev = at::kCUDA;
//         *dev = torch::kCUDA;
//         torch::Device device(torch::kCUDA, 1);
        
//         *dev = device.type();
// 	}
// 	else
// 	{
// 		//*dev = at::kCPU;
//         *dev = torch::kCPU;
// 	}
// 	// Change device to CPU by uncommenting the following line:
// 	// *dev = at::kCPU;
// 	log() << "current optimization device: " << *dev << std::endl;
//     logeol();
// 	return dev;
// }
//产生的mask的尺寸为：(l*x*y,alltwopinnet_num*pattern_num)
// void GlobalRouter::create_masks_fixed_agian(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *All_Gcell_Mask, c10::Device device)//这是针对四维的P矩阵使用的mask
// {
// 	auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
//     log() << "start updating masks" << std::endl;
//     vector<int> location_layer_index;
//     vector<int> location_x_index;
//     vector<int> location_y_index;
//     // vector<int> location_net_index;
//     // vector<int> location_tree_index;
//     vector<int> location_two_pin_net_index;
//     vector<int> location_pattern_index;


//     // int net_num=Two_Pin_net_vector.size();
//     // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
//     // int Max_tree_num_in_single_net=0;
//     // int Max_twopinnet_num_in_single_tree=0;
//     // for (const auto& net : Two_Pin_net_vector){
//     //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
//     //     for (const auto& tree : net){
//     //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
//     //     }
//     // }
//     int two_pin_net_num=Two_Pin_net_vector.size();
//     int pattern_num=Patterns.size();


//     for (int net_index=0; net_index<two_pin_net_num; net_index++){
//         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
//         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
//         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

//         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
//         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
//         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;

        
//         for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
//             int first_step_layerindex=Patterns[pattern_index][0];
//             int second_step_layerindex=Patterns[pattern_index][1];

//             //第一段原地穿孔
//             for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
//                 location_layer_index.push_back(layer_index);
//                 location_x_index.push_back(pin1_x_index);
//                 location_y_index.push_back(pin1_y_index);
//                 location_two_pin_net_index.push_back(net_index);
//                 location_pattern_index.push_back(pattern_index);
//             }
//             //第二段，由first_layer_index决定是先横着走还是先竖着走
//             if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
//                 for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
//                     location_layer_index.push_back(first_step_layerindex);
//                     location_x_index.push_back(x_index);
//                     location_y_index.push_back(pin1_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                     location_layer_index.push_back(layer_index);
//                     location_x_index.push_back(pin2_x_index);
//                     location_y_index.push_back(pin1_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.push_back(second_step_layerindex);
//                     location_x_index.push_back(pin2_x_index);
//                     location_y_index.push_back(y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//             }
//             else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.push_back(first_step_layerindex);
//                     location_x_index.push_back(pin1_x_index);
//                     location_y_index.push_back(y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                     location_layer_index.push_back(layer_index);
//                     location_x_index.push_back(pin1_x_index);
//                     location_y_index.push_back(pin2_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
//                     location_layer_index.push_back(second_step_layerindex);
//                     location_x_index.push_back(x_index);
//                     location_y_index.push_back(pin2_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
                    
//             }
//             for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
//                 location_layer_index.push_back(layer_index);
//                 location_x_index.push_back(pin2_x_index);
//                 location_y_index.push_back(pin2_y_index);
//                 location_two_pin_net_index.push_back(net_index);
//                 location_pattern_index.push_back(pattern_index);
//             }
//         }
//     }


//     vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
//     vector<vector<int>> multidimension_space_location_index;
//     multidimension_space_location_index.push_back(location_layer_index);
//     multidimension_space_location_index.push_back(location_x_index);
//     multidimension_space_location_index.push_back(location_y_index);
//     auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

    
//     vector<int> Parray_size={two_pin_net_num,pattern_num};
//     vector<vector<int>> multidimension_Parray_location_index;
//     multidimension_Parray_location_index.push_back(location_two_pin_net_index);
//     multidimension_Parray_location_index.push_back(location_pattern_index);
//     auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


    
//     torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
//     torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
//     torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
//     // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
//     std::cout<<space_location_index_tensor.size(0)<<std::endl;
//     std::cout<<space_location_index_tensor.dtype()<<std::endl;
    
//     // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
//     // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
//     *All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,two_pin_net_num*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    
//     //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
//     //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
//     log() << "update mask complete!" << std::endl;


// 	return;

// }
// void GlobalRouter::create_masks_fixed_agian(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *All_Gcell_Mask, c10::Device device)//这是针对四维的P矩阵使用的mask
// {
// 	auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
//     log() << "start updating masks" << std::endl;
//     vector<int> location_layer_index;
//     vector<int> location_x_index;
//     vector<int> location_y_index;
//     // vector<int> location_net_index;
//     // vector<int> location_tree_index;
//     vector<int> location_two_pin_net_index;
//     vector<int> location_pattern_index;


//     // int net_num=Two_Pin_net_vector.size();
//     // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
//     // int Max_tree_num_in_single_net=0;
//     // int Max_twopinnet_num_in_single_tree=0;
//     // for (const auto& net : Two_Pin_net_vector){
//     //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
//     //     for (const auto& tree : net){
//     //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
//     //     }
//     // }
//     int two_pin_net_num=Two_Pin_net_vector.size();
//     int pattern_num=Patterns.size();


//     for (int net_index=0; net_index<two_pin_net_num; net_index++){
//         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
//         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
//         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

//         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
//         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
//         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
//         for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
//             int first_step_layerindex=Patterns[pattern_index][0];
//             int second_step_layerindex=Patterns[pattern_index][1];

//             //第一段原地穿孔
//             // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin1_x_index);
//             //     location_y_index.push_back(pin1_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//             //第二段，由first_layer_index决定是先横着走还是先竖着走
//             if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
//                 for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
//                     location_layer_index.emplace_back(first_step_layerindex);
//                     location_x_index.emplace_back(x_index);
//                     location_y_index.emplace_back(pin1_y_index);
//                     location_two_pin_net_index.emplace_back(net_index);
//                     location_pattern_index.emplace_back(pattern_index);
//                 }
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin2_x_index);
//                 //     location_y_index.push_back(pin1_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.emplace_back(second_step_layerindex);
//                     location_x_index.emplace_back(pin2_x_index);
//                     location_y_index.emplace_back(y_index);
//                     location_two_pin_net_index.emplace_back(net_index);
//                     location_pattern_index.emplace_back(pattern_index);
//                 }
//             }
//             else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.emplace_back(first_step_layerindex);
//                     location_x_index.emplace_back(pin1_x_index);
//                     location_y_index.emplace_back(y_index);
//                     location_two_pin_net_index.emplace_back(net_index);
//                     location_pattern_index.emplace_back(pattern_index);
//                 }
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin1_x_index);
//                 //     location_y_index.push_back(pin2_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
//                     location_layer_index.emplace_back(second_step_layerindex);
//                     location_x_index.emplace_back(x_index);
//                     location_y_index.emplace_back(pin2_y_index);
//                     location_two_pin_net_index.emplace_back(net_index);
//                     location_pattern_index.emplace_back(pattern_index);
//                 }
                    
//             }
//             // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin2_x_index);
//             //     location_y_index.push_back(pin2_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//         }
//     }


//     vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
//     vector<vector<int>> multidimension_space_location_index;
//     multidimension_space_location_index.push_back(location_layer_index);
//     multidimension_space_location_index.push_back(location_x_index);
//     multidimension_space_location_index.push_back(location_y_index);
//     auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

    
//     vector<int> Parray_size={two_pin_net_num,pattern_num};
//     vector<vector<int>> multidimension_Parray_location_index;
//     multidimension_Parray_location_index.push_back(location_two_pin_net_index);
//     multidimension_Parray_location_index.push_back(location_pattern_index);
//     auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


    
//     torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
//     torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
//     torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
//     // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
//     std::cout<<space_location_index_tensor.size(0)<<std::endl;
//     std::cout<<space_location_index_tensor.dtype()<<std::endl;
    
//     // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
//     // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
//     *All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,two_pin_net_num*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    
//     //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
//     //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
//     log() << "update mask complete!" << std::endl;


// 	return;

// }

// //reserve+pushback
// void GlobalRouter::create_masks_fixed_agian(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *All_Gcell_Mask, c10::Device device)//这是针对四维的P矩阵使用的mask
// {
// 	auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
//     log() << "start updating masks" << std::endl;
//     vector<int> location_layer_index;
//     vector<int> location_x_index;
//     vector<int> location_y_index;
//     // vector<int> location_net_index;
//     // vector<int> location_tree_index;
//     vector<int> location_two_pin_net_index;
//     vector<int> location_pattern_index;


//     // int net_num=Two_Pin_net_vector.size();
//     // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
//     // int Max_tree_num_in_single_net=0;
//     // int Max_twopinnet_num_in_single_tree=0;
//     // for (const auto& net : Two_Pin_net_vector){
//     //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
//     //     for (const auto& tree : net){
//     //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
//     //     }
//     // }
//     int two_pin_net_num=Two_Pin_net_vector.size();
//     int pattern_num=Patterns.size();

//     int location_length=0;

    
//     for (int net_index=0; net_index<two_pin_net_num; net_index++){
//         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
//         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
//         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

//         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
//         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
//         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
//         for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
//             int first_step_layerindex=Patterns[pattern_index][0];
//             int second_step_layerindex=Patterns[pattern_index][1];

//             //第一段原地穿孔
//             // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin1_x_index);
//             //     location_y_index.push_back(pin1_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//             // location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex)+1;
//             //第二段，由first_layer_index决定是先横着走还是先竖着走
//             if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
//                 // for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
//                 //     location_layer_index.emplace_back(first_step_layerindex);
//                 //     location_x_index.emplace_back(x_index);
//                 //     location_y_index.emplace_back(pin1_y_index);
//                 //     location_two_pin_net_index.emplace_back(net_index);
//                 //     location_pattern_index.emplace_back(pattern_index);
//                 // }
//                 location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index)+1;
                
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin2_x_index);
//                 //     location_y_index.push_back(pin1_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 // location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex)+1;

//                 // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                 //     location_layer_index.emplace_back(second_step_layerindex);
//                 //     location_x_index.emplace_back(pin2_x_index);
//                 //     location_y_index.emplace_back(y_index);
//                 //     location_two_pin_net_index.emplace_back(net_index);
//                 //     location_pattern_index.emplace_back(pattern_index);
//                 // }
//                 location_length+=max(pin2_y_index,pin1_y_index)-min(pin2_y_index,pin1_y_index)+1;
//             }
//             else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
//                 // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                 //     location_layer_index.emplace_back(first_step_layerindex);
//                 //     location_x_index.emplace_back(pin1_x_index);
//                 //     location_y_index.emplace_back(y_index);
//                 //     location_two_pin_net_index.emplace_back(net_index);
//                 //     location_pattern_index.emplace_back(pattern_index);
//                 // }
//                 location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index)+1;
                
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin1_x_index);
//                 //     location_y_index.push_back(pin2_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 // location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex)+1;
//                 // for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
//                 //     location_layer_index.emplace_back(second_step_layerindex);
//                 //     location_x_index.emplace_back(x_index);
//                 //     location_y_index.emplace_back(pin2_y_index);
//                 //     location_two_pin_net_index.emplace_back(net_index);
//                 //     location_pattern_index.emplace_back(pattern_index);
//                 // }
//                 location_length+=max(pin2_x_index,pin1_x_index)-min(pin2_x_index,pin1_x_index)+1;
                    
//             }
//             // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin2_x_index);
//             //     location_y_index.push_back(pin2_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//             // location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex)+1;
//         }
//     }

//     location_layer_index.reserve(location_length);
//     location_x_index.reserve(location_length);
//     location_y_index.reserve(location_length);
//     //location_net_index.reserve(location_length);
//     //location_tree_index.reserve(location_length);
//     location_two_pin_net_index.reserve(location_length);
//     location_pattern_index.reserve(location_length);


//     for (int net_index=0; net_index<two_pin_net_num; net_index++){
//         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
//         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
//         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

//         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
//         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
//         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
//         for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
//             int first_step_layerindex=Patterns[pattern_index][0];
//             int second_step_layerindex=Patterns[pattern_index][1];

//             //第一段原地穿孔
//             // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin1_x_index);
//             //     location_y_index.push_back(pin1_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//             //第二段，由first_layer_index决定是先横着走还是先竖着走
//             if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
//                 for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
//                     location_layer_index.push_back(first_step_layerindex);
//                     location_x_index.push_back(x_index);
//                     location_y_index.push_back(pin1_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin2_x_index);
//                 //     location_y_index.push_back(pin1_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.push_back(second_step_layerindex);
//                     location_x_index.push_back(pin2_x_index);
//                     location_y_index.push_back(y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//             }
//             else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
//                 for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
//                     location_layer_index.push_back(first_step_layerindex);
//                     location_x_index.push_back(pin1_x_index);
//                     location_y_index.push_back(y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
//                 // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
//                 //     location_layer_index.push_back(layer_index);
//                 //     location_x_index.push_back(pin1_x_index);
//                 //     location_y_index.push_back(pin2_y_index);
//                 //     location_two_pin_net_index.push_back(net_index);
//                 //     location_pattern_index.push_back(pattern_index);
//                 // }
//                 for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
//                     location_layer_index.push_back(second_step_layerindex);
//                     location_x_index.push_back(x_index);
//                     location_y_index.push_back(pin2_y_index);
//                     location_two_pin_net_index.push_back(net_index);
//                     location_pattern_index.push_back(pattern_index);
//                 }
                    
//             }
//             // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
//             //     location_layer_index.push_back(layer_index);
//             //     location_x_index.push_back(pin2_x_index);
//             //     location_y_index.push_back(pin2_y_index);
//             //     location_two_pin_net_index.push_back(net_index);
//             //     location_pattern_index.push_back(pattern_index);
//             // }
//         }
//     }


//     vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
//     vector<vector<int>> multidimension_space_location_index;
//     multidimension_space_location_index.push_back(location_layer_index);
//     multidimension_space_location_index.push_back(location_x_index);
//     multidimension_space_location_index.push_back(location_y_index);
//     auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

    
//     vector<int> Parray_size={two_pin_net_num,pattern_num};
//     vector<vector<int>> multidimension_Parray_location_index;
//     multidimension_Parray_location_index.push_back(location_two_pin_net_index);
//     multidimension_Parray_location_index.push_back(location_pattern_index);
//     auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


    
//     torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
//     torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
//     torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
//     // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
//     std::cout<<space_location_index_tensor.size(0)<<std::endl;
//     std::cout<<space_location_index_tensor.dtype()<<std::endl;
    
//     // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
//     // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
//     // std::cout<<"thisok"<<std::endl;
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
//     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
//     *All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,two_pin_net_num*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    
//     //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
//     //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
//     log() << "update mask complete!" << std::endl;


// 	return;

// }


//resize+赋值+兼容了直线布线+产生wirelengthmask以及viacountmask
// void GlobalRouter::create_masks_fixed_agian_batch(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,vector<torch::Tensor> *All_Gcell_Mask,torch::Tensor *Wirelength_Mask,torch::Tensor *Viacount_Mask, c10::Device device,int batch_size)//这是针对四维的P矩阵使用的mask
void GlobalRouter::create_masks_fixed_agian_batch(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *All_Gcell_Mask,torch::Tensor *Wirelength_Mask,torch::Tensor *Viacount_Mask, c10::Device device)//这是针对四维的P矩阵使用的mask
{
	auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
    log() << "start updating masks" << std::endl;
    vector<int> location_layer_index;
    vector<int> location_x_index;
    vector<int> location_y_index;
    // vector<int> location_net_index;
    // vector<int> location_tree_index;
    vector<int> location_two_pin_net_index;
    vector<int> location_pattern_index;
    vector<float> pass_through_demand;


    // vector<float> wirelength_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);
    vector<int> turning_points_count_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);



    // int net_num=Two_Pin_net_vector.size();
    // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
    // int Max_tree_num_in_single_net=0;
    // int Max_twopinnet_num_in_single_tree=0;
    // for (const auto& net : Two_Pin_net_vector){
    //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
    //     for (const auto& tree : net){
    //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
    //     }
    // }
    int two_pin_net_num=Two_Pin_net_vector.size();
    int pattern_num=Patterns.size();

    int location_length=0;

    
    for (int net_index=0; net_index<two_pin_net_num; net_index++){
        auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
        auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
        auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

        auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
        auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
        auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
        //assert(!((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)));
        if((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)) std::cout<<"this is not a two pin net"<<std::endl;
        if((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
                int second_step_layerindex=Patterns[pattern_index][1];

                int one_step_layerindex=pin1_y_index==pin2_y_index?
                                        (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
                                        (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

                //先是原地穿孔
                location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);
                location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);

                //第二段，由one_step_layerindex决定是先横着走还是先竖着走
                //若是要横向走
                if(one_step_layerindex%2==0)location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
                //否则是要纵向走
                else location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);


                //最后还是原地穿孔
                location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
                location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
            }
        }
        else{//斜线情况
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];
                
                int second_step_layerindex=Patterns[pattern_index][1];

                //第一段原地穿孔
                // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin1_x_index);
                //     location_y_index.push_back(pin1_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                // }
                location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
                location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
                //第二段，由first_layer_index决定是先横着走还是先竖着走
                if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
                    // for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
                    //     location_layer_index.emplace_back(first_step_layerindex);
                    //     location_x_index.emplace_back(x_index);
                    //     location_y_index.emplace_back(pin1_y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
                    
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin2_x_index);
                    //     location_y_index.push_back(pin1_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);

                    // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
                    //     location_layer_index.emplace_back(second_step_layerindex);
                    //     location_x_index.emplace_back(pin2_x_index);
                    //     location_y_index.emplace_back(y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin2_y_index,pin1_y_index)-min(pin2_y_index,pin1_y_index);
                }
                else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
                    // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
                    //     location_layer_index.emplace_back(first_step_layerindex);
                    //     location_x_index.emplace_back(pin1_x_index);
                    //     location_y_index.emplace_back(y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);
                    
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin1_x_index);
                    //     location_y_index.push_back(pin2_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    // for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
                    //     location_layer_index.emplace_back(second_step_layerindex);
                    //     location_x_index.emplace_back(x_index);
                    //     location_y_index.emplace_back(pin2_y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin2_x_index,pin1_x_index)-min(pin2_x_index,pin1_x_index);
                        
                }
                // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin2_x_index);
                //     location_y_index.push_back(pin2_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                // }
                location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
                location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
            }
        }
    }

    location_layer_index.resize(location_length);
    location_x_index.resize(location_length);
    location_y_index.resize(location_length);
    //location_net_index.resize(location_length);
    //location_tree_index.resize(location_length);
    location_two_pin_net_index.resize(location_length);
    location_pattern_index.resize(location_length);
    pass_through_demand.resize(location_length);
    int location_count = 0;


    for (int net_index=0; net_index<two_pin_net_num; net_index++){
        auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
        auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
        auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

        auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
        auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
        auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
        if ((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
                int second_step_layerindex=Patterns[pattern_index][1];

                int one_step_layerindex=pin1_y_index==pin2_y_index?
                                        (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
                                        (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

                //先是原地穿孔
                for(int layer_index=min(pin1_layer_index,one_step_layerindex);layer_index<max(pin1_layer_index,one_step_layerindex);layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index;
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    
                }
                
                //第二段，由one_step_layerindex决定是先横着走还是先竖着走
                //若是要横向走
                if(one_step_layerindex%2==0){
                    for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
                        location_layer_index[location_count]=one_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
                        // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
                    }
                }
                //否则是要纵向走
                else {
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=one_step_layerindex;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
                        // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
                    }
                }


                //最后还是原地穿孔
                for(int layer_index=min(pin2_layer_index,one_step_layerindex);layer_index<max(pin2_layer_index,one_step_layerindex);layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index;
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                }
            }
        }
        else{//twopinnet是斜线
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];
                int second_step_layerindex=Patterns[pattern_index][1];

                //第一段原地穿孔
                for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<max(pin1_layer_index,first_step_layerindex); layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index;
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin1_x_index);
                //     location_y_index.push_back(pin1_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                }
                //第二段，由first_layer_index决定是先横着走还是先竖着走
                if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
                    for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
                        location_layer_index[location_count]=first_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,x_index);
                    }
                    for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
                        //lowerLoc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        //Loc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
                    }
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin2_x_index);
                    //     location_y_index.push_back(pin1_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=second_step_layerindex;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,y_index);
                    }
                }
                else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=first_step_layerindex;
                        location_x_index[location_count]=pin1_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{pin1_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,y_index);
                    }
                    for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
                        //lowerLoc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        //Loc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin1_x_index;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
                    }
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin1_x_index);
                    //     location_y_index.push_back(pin2_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    for(int x_index=min(pin2_x_index,pin1_x_index); x_index<max(pin2_x_index,pin1_x_index); x_index++){
                        location_layer_index[location_count]=second_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{x_index,pin2_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,x_index);
                    }
                        
                }
                for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<max(pin2_layer_index,second_step_layerindex); layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index;
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin2_x_index);
                //     location_y_index.push_back(pin2_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                }
            }
        }
    }
    assert(location_count==location_length);
    std::cout<<"location_count:"<<location_count<<";location_length:"<<location_length<<std::endl;

    vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
    vector<vector<int>> multidimension_space_location_index(gridgraph_size.size(),vector<int>(location_length));
    // multidimension_space_location_index.push_back(location_layer_index);
    // multidimension_space_location_index.push_back(location_x_index);
    // multidimension_space_location_index.push_back(location_y_index);
    multidimension_space_location_index[0]=(location_layer_index);
    multidimension_space_location_index[1]=(location_x_index);
    multidimension_space_location_index[2]=(location_y_index);
    auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

    
    vector<int> Parray_size={two_pin_net_num,pattern_num};
    vector<vector<int>> multidimension_Parray_location_index(Parray_size.size(),vector<int>(location_length));
    // multidimension_Parray_location_index.push_back(location_two_pin_net_index);
    // multidimension_Parray_location_index.push_back(location_pattern_index);
    multidimension_Parray_location_index[0]=(location_two_pin_net_index);
    multidimension_Parray_location_index[1]=(location_pattern_index);
    auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


    
    torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
    torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
    torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
    // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
    // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
    // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
    // std::cout<<"thisok"<<std::endl;
    



    // std::cout<<Parray_location_index_tensor.dtype()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].dtype()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0]<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item().to<int>()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item().to<float>()<<std::endl;
    // //检查一致性：
    // for(int i =0;i<space_location_index.size();i++) if(space_location_index[i]!=space_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;
    // for(int i =0;i<Parray_location_index.size();i++) if(Parray_location_index[i]!=Parray_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;


    torch::Tensor value=torch::tensor(pass_through_demand,torch::kFloat);
    //torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
    std::cout<<space_location_index_tensor.size(0)<<std::endl;
    std::cout<<space_location_index_tensor.dtype()<<std::endl;
    std::cout<<location_index_tensor.dtype()<<std::endl;
    std::cout<<value.dtype()<<std::endl;
    
    // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
    // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
    // std::cout<<"thisok"<<std::endl;
    //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
    //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
    *All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,two_pin_net_num*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    
    // *Wirelength_Mask = torch::tensor(wirelength_of_all_two_pin_net).reshape({1,2*two_pin_net_num}).to(device);
    *Viacount_Mask = torch::tensor(turning_points_count_of_all_two_pin_net,torch::kFloat).reshape({1,2*two_pin_net_num}).to(device);

    std::cout<<All_Gcell_Mask->dtype()<<std::endl;
    std::cout<<Wirelength_Mask->dtype()<<std::endl;
    std::cout<<Viacount_Mask->dtype()<<std::endl;
    //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
    log() << "update mask complete!" << std::endl;


	return;
    // auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
    // log() << "start updating masks" << std::endl;
    // vector<int> location_layer_index;
    // vector<int> location_x_index;
    // vector<int> location_y_index;
    // // vector<int> location_net_index;
    // // vector<int> location_tree_index;
    // vector<int> location_two_pin_net_index;
    // vector<int> location_pattern_index;
    // vector<float> pass_through_demand;


    // // vector<float> wirelength_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);
    // vector<int> turning_points_count_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);



    // // int net_num=Two_Pin_net_vector.size();
    // // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
    // // int Max_tree_num_in_single_net=0;
    // // int Max_twopinnet_num_in_single_tree=0;
    // // for (const auto& net : Two_Pin_net_vector){
    // //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
    // //     for (const auto& tree : net){
    // //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
    // //     }
    // // }
    // int two_pin_net_num=Two_Pin_net_vector.size();
    // int pattern_num=Patterns.size();

    

    // //int batch_num=(two_pin_net_num%batch_size)>0?two_pin_net_num/batch_size+1:two_pin_net_num/batch_size;
    // int batch_num=ceil(two_pin_net_num/batch_size);
    // std::cout<<"how many batch:"<<batch_num;
    // for(int batch_index=0;batch_index<batch_num;batch_index++){

    //     int location_length=0;
    //     for (int net_index=batch_index*batch_size; net_index<(batch_index+1)*batch_size; net_index++){
    //         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
    //         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
    //         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

    //         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
    //         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
    //         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
    //         //assert(!((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)));
    //         if((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)) std::cout<<"this is not a two pin net"<<std::endl;
    //         if((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
    //             for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
    //                 int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
    //                 int second_step_layerindex=Patterns[pattern_index][1];

    //                 int one_step_layerindex=pin1_y_index==pin2_y_index?
    //                                         (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
    //                                         (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

    //                 //先是原地穿孔
    //                 location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);
    //                 location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);

    //                 //第二段，由one_step_layerindex决定是先横着走还是先竖着走
    //                 //若是要横向走
    //                 if(one_step_layerindex%2==0)location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
    //                 //否则是要纵向走
    //                 else location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);


    //                 //最后还是原地穿孔
    //                 location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
    //                 location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
    //             }
    //         }
    //         else{//斜线情况
    //             for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
    //                 int first_step_layerindex=Patterns[pattern_index][0];
                    
    //                 int second_step_layerindex=Patterns[pattern_index][1];

    //                 //第一段原地穿孔
    //                 // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
    //                 //     location_layer_index.push_back(layer_index);
    //                 //     location_x_index.push_back(pin1_x_index);
    //                 //     location_y_index.push_back(pin1_y_index);
    //                 //     location_two_pin_net_index.push_back(net_index);
    //                 //     location_pattern_index.push_back(pattern_index);
    //                 // }
    //                 location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
    //                 location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
    //                 //第二段，由first_layer_index决定是先横着走还是先竖着走
    //                 if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
    //                     // for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
    //                     //     location_layer_index.emplace_back(first_step_layerindex);
    //                     //     location_x_index.emplace_back(x_index);
    //                     //     location_y_index.emplace_back(pin1_y_index);
    //                     //     location_two_pin_net_index.emplace_back(net_index);
    //                     //     location_pattern_index.emplace_back(pattern_index);
    //                     // }
    //                     location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
                        
    //                     // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                     //     location_layer_index.push_back(layer_index);
    //                     //     location_x_index.push_back(pin2_x_index);
    //                     //     location_y_index.push_back(pin1_y_index);
    //                     //     location_two_pin_net_index.push_back(net_index);
    //                     //     location_pattern_index.push_back(pattern_index);
    //                     // }
    //                     location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
    //                     location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);

    //                     // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
    //                     //     location_layer_index.emplace_back(second_step_layerindex);
    //                     //     location_x_index.emplace_back(pin2_x_index);
    //                     //     location_y_index.emplace_back(y_index);
    //                     //     location_two_pin_net_index.emplace_back(net_index);
    //                     //     location_pattern_index.emplace_back(pattern_index);
    //                     // }
    //                     location_length+=max(pin2_y_index,pin1_y_index)-min(pin2_y_index,pin1_y_index);
    //                 }
    //                 else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
    //                     // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
    //                     //     location_layer_index.emplace_back(first_step_layerindex);
    //                     //     location_x_index.emplace_back(pin1_x_index);
    //                     //     location_y_index.emplace_back(y_index);
    //                     //     location_two_pin_net_index.emplace_back(net_index);
    //                     //     location_pattern_index.emplace_back(pattern_index);
    //                     // }
    //                     location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);
                        
    //                     // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                     //     location_layer_index.push_back(layer_index);
    //                     //     location_x_index.push_back(pin1_x_index);
    //                     //     location_y_index.push_back(pin2_y_index);
    //                     //     location_two_pin_net_index.push_back(net_index);
    //                     //     location_pattern_index.push_back(pattern_index);
    //                     // }
    //                     location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
    //                     location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
    //                     // for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
    //                     //     location_layer_index.emplace_back(second_step_layerindex);
    //                     //     location_x_index.emplace_back(x_index);
    //                     //     location_y_index.emplace_back(pin2_y_index);
    //                     //     location_two_pin_net_index.emplace_back(net_index);
    //                     //     location_pattern_index.emplace_back(pattern_index);
    //                     // }
    //                     location_length+=max(pin2_x_index,pin1_x_index)-min(pin2_x_index,pin1_x_index);
                            
    //                 }
    //                 // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
    //                 //     location_layer_index.push_back(layer_index);
    //                 //     location_x_index.push_back(pin2_x_index);
    //                 //     location_y_index.push_back(pin2_y_index);
    //                 //     location_two_pin_net_index.push_back(net_index);
    //                 //     location_pattern_index.push_back(pattern_index);
    //                 // }
    //                 location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
    //                 location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
    //             }
    //         }
    //     }

    //     location_layer_index.resize(location_length);
    //     location_x_index.resize(location_length);
    //     location_y_index.resize(location_length);
    //     //location_net_index.resize(location_length);
    //     //location_tree_index.resize(location_length);
    //     location_two_pin_net_index.resize(location_length);
    //     location_pattern_index.resize(location_length);
    //     pass_through_demand.resize(location_length);
    //     int location_count = 0;


    //     for (int net_index=batch_index*batch_size; net_index<(batch_index+1)*batch_size; net_index++){
    //         auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
    //         auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
    //         auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

    //         auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
    //         auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
    //         auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
    //         if ((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
    //             for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
    //                 int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
    //                 int second_step_layerindex=Patterns[pattern_index][1];

    //                 int one_step_layerindex=pin1_y_index==pin2_y_index?
    //                                         (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
    //                                         (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

    //                 //先是原地穿孔
    //                 for(int layer_index=min(pin1_layer_index,one_step_layerindex);layer_index<max(pin1_layer_index,one_step_layerindex);layer_index++){
    //                     //lowerLoc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
    //                     location_y_index[location_count]=pin1_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;

    //                     //Loc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin1_x_index;
    //                     location_y_index[location_count]=pin1_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;

                        
    //                 }
                    
    //                 //第二段，由one_step_layerindex决定是先横着走还是先竖着走
    //                 //若是要横向走
    //                 if(one_step_layerindex%2==0){
    //                     for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
    //                         location_layer_index[location_count]=one_step_layerindex;
    //                         location_x_index[location_count]=x_index;
    //                         location_y_index[location_count]=pin1_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
    //                     }
    //                 }
    //                 //否则是要纵向走
    //                 else {
    //                     for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
    //                         location_layer_index[location_count]=one_step_layerindex;
    //                         location_x_index[location_count]=pin2_x_index;
    //                         location_y_index[location_count]=y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
    //                     }
    //                 }


    //                 //最后还是原地穿孔
    //                 for(int layer_index=min(pin2_layer_index,one_step_layerindex);layer_index<max(pin2_layer_index,one_step_layerindex);layer_index++){
    //                     //lowerLoc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
    //                     location_y_index[location_count]=pin2_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;

    //                     //Loc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin2_x_index;
    //                     location_y_index[location_count]=pin2_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;
    //                 }
    //             }
    //         }
    //         else{//twopinnet是斜线
    //             for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
    //                 int first_step_layerindex=Patterns[pattern_index][0];
    //                 int second_step_layerindex=Patterns[pattern_index][1];

    //                 //第一段原地穿孔
    //                 for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<max(pin1_layer_index,first_step_layerindex); layer_index++){
    //                     //lowerLoc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
    //                     location_y_index[location_count]=pin1_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;

    //                     //Loc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin1_x_index;
    //                     location_y_index[location_count]=pin1_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;
    //                 //     location_layer_index.push_back(layer_index);
    //                 //     location_x_index.push_back(pin1_x_index);
    //                 //     location_y_index.push_back(pin1_y_index);
    //                 //     location_two_pin_net_index.push_back(net_index);
    //                 //     location_pattern_index.push_back(pattern_index);
    //                 }
    //                 //第二段，由first_layer_index决定是先横着走还是先竖着走
    //                 if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
    //                     for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
    //                         location_layer_index[location_count]=first_step_layerindex;
    //                         location_x_index[location_count]=x_index;
    //                         location_y_index[location_count]=pin1_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,x_index);
    //                     }
    //                     for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                         //lowerLoc
    //                         location_layer_index[location_count]=layer_index;
    //                         location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
    //                         location_y_index[location_count]=pin1_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
    //                         //pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         //Loc
    //                         location_layer_index[location_count]=layer_index;
    //                         location_x_index[location_count]=pin2_x_index;
    //                         location_y_index[location_count]=pin1_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
    //                         //pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
    //                     }
    //                     // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                     //     location_layer_index.push_back(layer_index);
    //                     //     location_x_index.push_back(pin2_x_index);
    //                     //     location_y_index.push_back(pin1_y_index);
    //                     //     location_two_pin_net_index.push_back(net_index);
    //                     //     location_pattern_index.push_back(pattern_index);
    //                     // }
    //                     for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
    //                         location_layer_index[location_count]=second_step_layerindex;
    //                         location_x_index[location_count]=pin2_x_index;
    //                         location_y_index[location_count]=y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,y_index);
    //                     }
    //                 }
    //                 else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
    //                     for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
    //                         location_layer_index[location_count]=first_step_layerindex;
    //                         location_x_index[location_count]=pin1_x_index;
    //                         location_y_index[location_count]=y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{pin1_x_index,y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,y_index);
    //                     }
    //                     for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                         //lowerLoc
    //                         location_layer_index[location_count]=layer_index;
    //                         location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;
    //                         location_y_index[location_count]=pin2_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
    //                         //pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         //Loc
    //                         location_layer_index[location_count]=layer_index;
    //                         location_x_index[location_count]=pin1_x_index;
    //                         location_y_index[location_count]=pin2_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
    //                         //pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
    //                     }
    //                     // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
    //                     //     location_layer_index.push_back(layer_index);
    //                     //     location_x_index.push_back(pin1_x_index);
    //                     //     location_y_index.push_back(pin2_y_index);
    //                     //     location_two_pin_net_index.push_back(net_index);
    //                     //     location_pattern_index.push_back(pattern_index);
    //                     // }
    //                     for(int x_index=min(pin2_x_index,pin1_x_index); x_index<max(pin2_x_index,pin1_x_index); x_index++){
    //                         location_layer_index[location_count]=second_step_layerindex;
    //                         location_x_index[location_count]=x_index;
    //                         location_y_index[location_count]=pin2_y_index;
    //                         location_two_pin_net_index[location_count]=net_index;
    //                         location_pattern_index[location_count]=pattern_index;
    //                         //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{x_index,pin2_y_index});
    //                         pass_through_demand[location_count]=1;
    //                         location_count++;

    //                         // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,x_index);
    //                     }
                            
    //                 }
    //                 for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<max(pin2_layer_index,second_step_layerindex); layer_index++){
    //                     //lowerLoc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
    //                     location_y_index[location_count]=pin2_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;

    //                     //Loc
    //                     location_layer_index[location_count]=layer_index;
    //                     location_x_index[location_count]=pin2_x_index;
    //                     location_y_index[location_count]=pin2_y_index;
    //                     location_two_pin_net_index[location_count]=net_index;
    //                     location_pattern_index[location_count]=pattern_index;
    //                     pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
    //                     //pass_through_demand[location_count]=1;
    //                     location_count++;
    //                 //     location_layer_index.push_back(layer_index);
    //                 //     location_x_index.push_back(pin2_x_index);
    //                 //     location_y_index.push_back(pin2_y_index);
    //                 //     location_two_pin_net_index.push_back(net_index);
    //                 //     location_pattern_index.push_back(pattern_index);
    //                 }
    //             }
    //         }
    //     }
    //     assert(location_count==location_length);
    //     // std::cout<<"location_count:"<<location_count<<";location_length:"<<location_length<<std::endl;

    //     vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
    //     vector<vector<int>> multidimension_space_location_index(gridgraph_size.size(),vector<int>(location_length));
    //     // multidimension_space_location_index.push_back(location_layer_index);
    //     // multidimension_space_location_index.push_back(location_x_index);
    //     // multidimension_space_location_index.push_back(location_y_index);
    //     multidimension_space_location_index[0]=(location_layer_index);
    //     multidimension_space_location_index[1]=(location_x_index);
    //     multidimension_space_location_index[2]=(location_y_index);
    //     auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

        
    //     vector<int> Parray_size={batch_size,pattern_num};
    //     vector<vector<int>> multidimension_Parray_location_index(Parray_size.size(),vector<int>(location_length));
    //     // multidimension_Parray_location_index.push_back(location_two_pin_net_index);
    //     // multidimension_Parray_location_index.push_back(location_pattern_index);
    //     multidimension_Parray_location_index[0]=(location_two_pin_net_index);
    //     multidimension_Parray_location_index[1]=(location_pattern_index);
    //     auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


        
    //     torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
    //     torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
    //     torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
    //     // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
    //     // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
    //     // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
    //     // std::cout<<"thisok"<<std::endl;
        



    //     // std::cout<<Parray_location_index_tensor.dtype()<<std::endl;
    //     // std::cout<<Parray_location_index_tensor[0].dtype()<<std::endl;
    //     // std::cout<<Parray_location_index_tensor[0]<<std::endl;
    //     // std::cout<<Parray_location_index_tensor[0].item()<<std::endl;
    //     // std::cout<<Parray_location_index_tensor[0].item().to<int>()<<std::endl;
    //     // std::cout<<Parray_location_index_tensor[0].item().to<float>()<<std::endl;
    //     // //检查一致性：
    //     // for(int i =0;i<space_location_index.size();i++) if(space_location_index[i]!=space_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;
    //     // for(int i =0;i<Parray_location_index.size();i++) if(Parray_location_index[i]!=Parray_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;


    //     torch::Tensor value=torch::tensor(pass_through_demand,torch::kFloat);
    //     //torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
    //     // std::cout<<space_location_index_tensor.size(0)<<std::endl;
    //     // std::cout<<space_location_index_tensor.dtype()<<std::endl;
    //     // std::cout<<location_index_tensor.dtype()<<std::endl;
    //     // std::cout<<value.dtype()<<std::endl;
        
    //     // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
    //     // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
    //     // std::cout<<"thisok"<<std::endl;
    //     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
    //     //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
    //     (*All_Gcell_Mask)[batch_index] = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,batch_size*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
        
    //     // *Wirelength_Mask = torch::tensor(wirelength_of_all_two_pin_net).reshape({1,2*two_pin_net_num}).to(device);
    //     *Viacount_Mask = torch::tensor(turning_points_count_of_all_two_pin_net,torch::kFloat).index({torch::indexing::Slice(batch_index*batch_size*pattern_num,(batch_index+1)*batch_size*pattern_num)}).reshape({1,batch_size*pattern_num}).to(device);
        
    // }
    // // std::cout<<All_Gcell_Mask->dtype()<<std::endl;
    // // std::cout<<Wirelength_Mask->dtype()<<std::endl;
    // // std::cout<<Viacount_Mask->dtype()<<std::endl;
    // //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    // //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
    // log() << "update mask complete!" << std::endl;


	// return;

}

//resize+赋值+兼容了直线布线+产生wirelengthmask以及viacountmask
void GlobalRouter::create_masks_fixed_agian(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y,vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *All_Gcell_Mask,torch::Tensor *Wirelength_Mask,torch::Tensor *Viacount_Mask, c10::Device device)//这是针对四维的P矩阵使用的mask
{
	auto Patterns=linkBetweenSelectedPatternLayerAndLPatternIndex(Layer_Num);
    
    log() << "start updating masks" << std::endl;
    vector<int> location_layer_index;
    vector<int> location_x_index;
    vector<int> location_y_index;
    // vector<int> location_net_index;
    // vector<int> location_tree_index;
    vector<int> location_two_pin_net_index;
    vector<int> location_pattern_index;
    vector<float> pass_through_demand;


    // vector<float> wirelength_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);
    vector<int> turning_points_count_of_all_two_pin_net(2*Two_Pin_net_vector.size(),0);



    // int net_num=Two_Pin_net_vector.size();
    // //以下一段可以获取所有net中最大的tree_num和所有tree中最大的twopinnet_num
    // int Max_tree_num_in_single_net=0;
    // int Max_twopinnet_num_in_single_tree=0;
    // for (const auto& net : Two_Pin_net_vector){
    //     Max_tree_num_in_single_net = std::max(Max_tree_num_in_single_net,static_cast<int>(net.size()));
    //     for (const auto& tree : net){
    //         Max_twopinnet_num_in_single_tree = std::max(Max_twopinnet_num_in_single_tree,static_cast<int>(tree.size()));
    //     }
    // }
    int two_pin_net_num=Two_Pin_net_vector.size();
    int pattern_num=Patterns.size();

    int location_length=0;

    
    for (int net_index=0; net_index<two_pin_net_num; net_index++){
        auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
        auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
        auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

        auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
        auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
        auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
        //assert(!((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)));
        if((pin1_x_index==pin2_x_index)&&(pin1_y_index==pin2_y_index)) std::cout<<"this is not a two pin net"<<std::endl;
        if((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
                int second_step_layerindex=Patterns[pattern_index][1];

                int one_step_layerindex=pin1_y_index==pin2_y_index?
                                        (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
                                        (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

                //先是原地穿孔
                location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);
                location_length+=max(pin1_layer_index,one_step_layerindex)-min(pin1_layer_index,one_step_layerindex);

                //第二段，由one_step_layerindex决定是先横着走还是先竖着走
                //若是要横向走
                if(one_step_layerindex%2==0)location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
                //否则是要纵向走
                else location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);


                //最后还是原地穿孔
                location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
                location_length+=max(pin2_layer_index,one_step_layerindex)-min(pin2_layer_index,one_step_layerindex);
            }
        }
        else{//斜线情况
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];
                
                int second_step_layerindex=Patterns[pattern_index][1];

                //第一段原地穿孔
                // for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<=max(pin1_layer_index,first_step_layerindex); layer_index++){
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin1_x_index);
                //     location_y_index.push_back(pin1_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                // }
                location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
                location_length+=max(pin1_layer_index,first_step_layerindex)-min(pin1_layer_index,first_step_layerindex);
                //第二段，由first_layer_index决定是先横着走还是先竖着走
                if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
                    // for(int x_index=min(pin1_x_index,pin2_x_index); x_index<=max(pin1_x_index,pin2_x_index); x_index++){
                    //     location_layer_index.emplace_back(first_step_layerindex);
                    //     location_x_index.emplace_back(x_index);
                    //     location_y_index.emplace_back(pin1_y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin1_x_index,pin2_x_index)-min(pin1_x_index,pin2_x_index);
                    
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin2_x_index);
                    //     location_y_index.push_back(pin1_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);

                    // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
                    //     location_layer_index.emplace_back(second_step_layerindex);
                    //     location_x_index.emplace_back(pin2_x_index);
                    //     location_y_index.emplace_back(y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin2_y_index,pin1_y_index)-min(pin2_y_index,pin1_y_index);
                }
                else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
                    // for(int y_index=min(pin2_y_index,pin1_y_index); y_index<=max(pin2_y_index,pin1_y_index); y_index++){
                    //     location_layer_index.emplace_back(first_step_layerindex);
                    //     location_x_index.emplace_back(pin1_x_index);
                    //     location_y_index.emplace_back(y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin1_y_index,pin2_y_index)-min(pin1_y_index,pin2_y_index);
                    
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin1_x_index);
                    //     location_y_index.push_back(pin2_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    location_length+=max(first_step_layerindex,second_step_layerindex)-min(first_step_layerindex,second_step_layerindex);
                    // for(int x_index=min(pin2_x_index,pin1_x_index); x_index<=max(pin2_x_index,pin1_x_index); x_index++){
                    //     location_layer_index.emplace_back(second_step_layerindex);
                    //     location_x_index.emplace_back(x_index);
                    //     location_y_index.emplace_back(pin2_y_index);
                    //     location_two_pin_net_index.emplace_back(net_index);
                    //     location_pattern_index.emplace_back(pattern_index);
                    // }
                    location_length+=max(pin2_x_index,pin1_x_index)-min(pin2_x_index,pin1_x_index);
                        
                }
                // for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<=max(pin2_layer_index,second_step_layerindex); layer_index++){
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin2_x_index);
                //     location_y_index.push_back(pin2_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                // }
                location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
                location_length+=max(pin2_layer_index,second_step_layerindex)-min(pin2_layer_index,second_step_layerindex);
            }
        }
    }

    location_layer_index.resize(location_length);
    location_x_index.resize(location_length);
    location_y_index.resize(location_length);
    //location_net_index.resize(location_length);
    //location_tree_index.resize(location_length);
    location_two_pin_net_index.resize(location_length);
    location_pattern_index.resize(location_length);
    pass_through_demand.resize(location_length);
    int location_count = 0;


    for (int net_index=0; net_index<two_pin_net_num; net_index++){
        auto pin1_layer_index=Two_Pin_net_vector[net_index].pin1.layer;
        auto pin1_x_index=Two_Pin_net_vector[net_index].pin1.x;
        auto pin1_y_index=Two_Pin_net_vector[net_index].pin1.y;

        auto pin2_layer_index=Two_Pin_net_vector[net_index].pin2.layer;
        auto pin2_x_index=Two_Pin_net_vector[net_index].pin2.x;
        auto pin2_y_index=Two_Pin_net_vector[net_index].pin2.y;
        if ((pin1_y_index==pin2_y_index)||(pin1_x_index==pin2_x_index)){//说明是直线走线的net
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];//在模型中仍然假定其是两步走的，只不过其中的一步走的长度是0
                int second_step_layerindex=Patterns[pattern_index][1];

                int one_step_layerindex=pin1_y_index==pin2_y_index?
                                        (first_step_layerindex%2==0?first_step_layerindex:second_step_layerindex)://若是horizonal（说明直线走线只能走偶数层），且firstlayer是偶数，说明这条线直线走到就是这个firstlayer
                                        (first_step_layerindex%2==1?first_step_layerindex:second_step_layerindex);

                //先是原地穿孔
                for(int layer_index=min(pin1_layer_index,one_step_layerindex);layer_index<max(pin1_layer_index,one_step_layerindex);layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index;
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    
                }
                
                //第二段，由one_step_layerindex决定是先横着走还是先竖着走
                //若是要横向走
                if(one_step_layerindex%2==0){
                    for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
                        location_layer_index[location_count]=one_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
                        // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,x_index);
                    }
                }
                //否则是要纵向走
                else {
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=one_step_layerindex;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
                        // wirelength_of_all_two_pin_net[net_index*pattern_num+1]+=gridGraph.getEdgeLength(one_step_layerindex%2,y_index);
                    }
                }


                //最后还是原地穿孔
                for(int layer_index=min(pin2_layer_index,one_step_layerindex);layer_index<max(pin2_layer_index,one_step_layerindex);layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index;
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                }
            }
        }
        else{//twopinnet是斜线
            for(int pattern_index=0; pattern_index<Patterns.size(); pattern_index++){
                int first_step_layerindex=Patterns[pattern_index][0];
                int second_step_layerindex=Patterns[pattern_index][1];

                //第一段原地穿孔
                for(int layer_index=min(pin1_layer_index,first_step_layerindex); layer_index<max(pin1_layer_index,first_step_layerindex); layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin1_x_index;
                    location_y_index[location_count]=pin1_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin1_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin1_x_index);
                //     location_y_index.push_back(pin1_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                }
                //第二段，由first_layer_index决定是先横着走还是先竖着走
                if(first_step_layerindex%2==0){//第一段是横向走线，说明要横向走，那么pattern的拐点在（x2，y1）
                    for(int x_index=min(pin1_x_index,pin2_x_index); x_index<max(pin1_x_index,pin2_x_index); x_index++){
                        location_layer_index[location_count]=first_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{x_index,pin1_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,x_index);
                    }
                    for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
                        //lowerLoc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        //Loc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=pin1_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin1_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
                    }
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin2_x_index);
                    //     location_y_index.push_back(pin1_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=second_step_layerindex;
                        location_x_index[location_count]=pin2_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{pin2_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,y_index);
                    }
                }
                else{//如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
                    for(int y_index=min(pin2_y_index,pin1_y_index); y_index<max(pin2_y_index,pin1_y_index); y_index++){
                        location_layer_index[location_count]=first_step_layerindex;
                        location_x_index[location_count]=pin1_x_index;
                        location_y_index[location_count]=y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(first_step_layerindex,{pin1_x_index,y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(first_step_layerindex,y_index);
                    }
                    for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<max(first_step_layerindex,second_step_layerindex); layer_index++){
                        //lowerLoc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin1_x_index>0?pin1_x_index-1:0;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        //Loc
                        location_layer_index[location_count]=layer_index;
                        location_x_index[location_count]=pin1_x_index;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin1_x_index,pin2_y_index});
                        //pass_through_demand[location_count]=1;
                        location_count++;

                        turning_points_count_of_all_two_pin_net[net_index*pattern_num+pattern_index]=1;
                    }
                    // for(int layer_index=min(first_step_layerindex,second_step_layerindex); layer_index<=max(first_step_layerindex,second_step_layerindex); layer_index++){
                    //     location_layer_index.push_back(layer_index);
                    //     location_x_index.push_back(pin1_x_index);
                    //     location_y_index.push_back(pin2_y_index);
                    //     location_two_pin_net_index.push_back(net_index);
                    //     location_pattern_index.push_back(pattern_index);
                    // }
                    for(int x_index=min(pin2_x_index,pin1_x_index); x_index<max(pin2_x_index,pin1_x_index); x_index++){
                        location_layer_index[location_count]=second_step_layerindex;
                        location_x_index[location_count]=x_index;
                        location_y_index[location_count]=pin2_y_index;
                        location_two_pin_net_index[location_count]=net_index;
                        location_pattern_index[location_count]=pattern_index;
                        //pass_through_demand[location_count]=gridGraph.getWiredemand_for_mask(second_step_layerindex,{x_index,pin2_y_index});
                        pass_through_demand[location_count]=1;
                        location_count++;

                        // wirelength_of_all_two_pin_net[net_index*pattern_num+pattern_index]+=gridGraph.getEdgeLength(second_step_layerindex,x_index);
                    }
                        
                }
                for(int layer_index=min(pin2_layer_index,second_step_layerindex); layer_index<max(pin2_layer_index,second_step_layerindex); layer_index++){
                    //lowerLoc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index>0?pin2_x_index-1:0;//因为只有两层，所以穿孔的较低层一定是在第0层的。如果后续改成了3D形式的，那这里一定需要变化。并且要注意这里的坐标如果本来是0的话，经过-1就小于0了，会导致cuda报错（这个害惨我了，找了老长时间）
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;

                    //Loc
                    location_layer_index[location_count]=layer_index;
                    location_x_index[location_count]=pin2_x_index;
                    location_y_index[location_count]=pin2_y_index;
                    location_two_pin_net_index[location_count]=net_index;
                    location_pattern_index[location_count]=pattern_index;
                    pass_through_demand[location_count]=gridGraph.getViademand_for_mask(layer_index,{pin2_x_index,pin2_y_index});
                    //pass_through_demand[location_count]=1;
                    location_count++;
                //     location_layer_index.push_back(layer_index);
                //     location_x_index.push_back(pin2_x_index);
                //     location_y_index.push_back(pin2_y_index);
                //     location_two_pin_net_index.push_back(net_index);
                //     location_pattern_index.push_back(pattern_index);
                }
            }
        }
    }
    assert(location_count==location_length);
    std::cout<<"location_count:"<<location_count<<";location_length:"<<location_length<<std::endl;

    vector<int> gridgraph_size={Layer_Num,Gcell_Num_X,Gcell_Num_Y};
    vector<vector<int>> multidimension_space_location_index(gridgraph_size.size(),vector<int>(location_length));
    // multidimension_space_location_index.push_back(location_layer_index);
    // multidimension_space_location_index.push_back(location_x_index);
    // multidimension_space_location_index.push_back(location_y_index);
    multidimension_space_location_index[0]=(location_layer_index);
    multidimension_space_location_index[1]=(location_x_index);
    multidimension_space_location_index[2]=(location_y_index);
    auto space_location_index=compress_multidimensional_index_to_one_dimensional_index(gridgraph_size,multidimension_space_location_index);

    
    vector<int> Parray_size={two_pin_net_num,pattern_num};
    vector<vector<int>> multidimension_Parray_location_index(Parray_size.size(),vector<int>(location_length));
    // multidimension_Parray_location_index.push_back(location_two_pin_net_index);
    // multidimension_Parray_location_index.push_back(location_pattern_index);
    multidimension_Parray_location_index[0]=(location_two_pin_net_index);
    multidimension_Parray_location_index[1]=(location_pattern_index);
    auto Parray_location_index=compress_multidimensional_index_to_one_dimensional_index(Parray_size,multidimension_Parray_location_index);


    
    torch::Tensor space_location_index_tensor=torch::tensor(space_location_index);//这里数据类型似乎也不能变
    torch::Tensor Parray_location_index_tensor=torch::tensor(Parray_location_index);
    torch::Tensor location_index_tensor=torch::stack({space_location_index_tensor,Parray_location_index_tensor});
    // for(int64_t dim:space_location_index_tensor.sizes()) std::cout<<dim<<" ";
    // for(int64_t dim:Parray_location_index_tensor.sizes()) std::cout<<dim<<" ";
    // for (int64_t dim : location_index_tensor.sizes()) std::cout<<dim<<" ";
    // std::cout<<"thisok"<<std::endl;
    



    // std::cout<<Parray_location_index_tensor.dtype()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].dtype()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0]<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item().to<int>()<<std::endl;
    // std::cout<<Parray_location_index_tensor[0].item().to<float>()<<std::endl;
    // //检查一致性：
    // for(int i =0;i<space_location_index.size();i++) if(space_location_index[i]!=space_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;
    // for(int i =0;i<Parray_location_index.size();i++) if(Parray_location_index[i]!=Parray_location_index_tensor[i].item().to<float>()) std::cout<<"have problem"<<std::endl;


    torch::Tensor value=torch::tensor(pass_through_demand,torch::kFloat);
    //torch::Tensor value=torch::ones(space_location_index_tensor.size(0));
    std::cout<<space_location_index_tensor.size(0)<<std::endl;
    std::cout<<space_location_index_tensor.dtype()<<std::endl;
    std::cout<<location_index_tensor.dtype()<<std::endl;
    std::cout<<value.dtype()<<std::endl;
    
    // std::cout<<location_index_tensor.size(0)<<location_index_tensor.size(1)<<value.size(0)<<std::endl;
    // for (int64_t dim : value.sizes()) std::cout<<dim<<" ";
    // std::cout<<"thisok"<<std::endl;
    //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,(Layer_Num*Gcell_Num_X*Gcell_Num_Y,edges.size()*Patterns.size()));
    //*All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor,value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,long(edges.size())*Patterns.size()});
    *All_Gcell_Mask = torch::sparse_coo_tensor(location_index_tensor, value,{Layer_Num*Gcell_Num_X*Gcell_Num_Y,two_pin_net_num*pattern_num}).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    
    // *Wirelength_Mask = torch::tensor(wirelength_of_all_two_pin_net).reshape({1,2*two_pin_net_num}).to(device);
    *Viacount_Mask = torch::tensor(turning_points_count_of_all_two_pin_net,torch::kFloat).reshape({1,2*two_pin_net_num}).to(device);

    std::cout<<All_Gcell_Mask->dtype()<<std::endl;
    std::cout<<Wirelength_Mask->dtype()<<std::endl;
    std::cout<<Viacount_Mask->dtype()<<std::endl;
    //*All_Gcell_Mask = (*All_Gcell_Mask).to(torch::kFloat).to(device);//这个地方的类型必须是kFloat，如果是kInt就会有问题
    //for (int64_t dim : *All_Gcell_Mask.sizes()) std::cout<<dim<<" ";
    log() << "update mask complete!" << std::endl;


	return;

}
std::vector<std::array<int, 2>> GlobalRouter::linkBetweenSelectedPatternLayerAndLPatternIndex(int layer_num) {
    int L_pattern_num;
    if (layer_num % 2) { // 余数为1，则层数为奇数
        L_pattern_num = int(layer_num * (layer_num - 1) / 2 - ((int(layer_num / 2) + 1) * int(layer_num / 2) / 2 + int(layer_num / 2) * (int(layer_num / 2) - 1) / 2)) * 2;
    } else { // 层数为偶数
        L_pattern_num = int((layer_num * (layer_num - 1) / 2 - layer_num / 2 * (layer_num / 2 - 1)) * 2);
    }
    std::vector<std::array<int, 2>> link_between_selectedpatternlayer_and_L_pattern_index(L_pattern_num, std::array<int, 2>{0, 0});
    std::vector<int> m0, m1;

    for (int x = 0; x < layer_num; ++x) {
        if (x % 2 == 0) m0.push_back(x);
        else m1.push_back(x);
    }

    int k = 0;
    for (int i : m0) {
        for (int j : m1) {
            link_between_selectedpatternlayer_and_L_pattern_index[k] = {i, j};
            ++k;
            link_between_selectedpatternlayer_and_L_pattern_index[k] = {j, i};
            ++k;
        }
    }

    return link_between_selectedpatternlayer_and_L_pattern_index;
}

vector<int> GlobalRouter::compress_multidimensional_index_to_one_dimensional_index(vector<int>& multidimension_maxvalue,vector<vector<int>>& multidimension_index) 
{
    vector<int> one_dimensional_index(multidimension_index[0].size());
    //one_dimensional_index.resize(multidimension_index[0].size());
    //if(multidimension_maxvalue.size() != multidimension_index[0].size() | multidimension_maxvalue.size() != multidimension_index[1].size()) std::cout<<"compress_multidimensional_index_to_one_dimensional_index_error"<<std::endl;
    if(multidimension_maxvalue.size() != multidimension_index.size()) std::cout<<"compress_multidimensional_index_to_one_dimensional_index_error"<<std::endl;
    else{
        for(int location_num=0; location_num<multidimension_index[0].size() ; location_num++)
        {
            int result=0;
            
            for(int dimension=0;dimension<multidimension_index.size();dimension++) result=result*multidimension_maxvalue[dimension]+multidimension_index[dimension][location_num];

            //one_dimensional_index.push_back(result);
            one_dimensional_index[location_num] = result;
        }
        return one_dimensional_index;
    }
    return one_dimensional_index;
}



torch::Tensor GlobalRouter::generate_sparse_tensor_based_on_multidimensional_index(std::vector<std::vector<int>>& Multidimension_Index,int Value_In_Sparse_Tensor,std::vector<int> Sparse_Tensor_Size){
    int dimension = Multidimension_Index.size();
    int sparse_point_num = Multidimension_Index[0].size();
    torch::Tensor index_tensor = torch::tensor(Multidimension_Index[0],torch::kLong).unsqueeze(0);
    auto value_tensor = torch::ones(sparse_point_num);
    for (int Dim=1;Dim<dimension;Dim++){
        index_tensor = torch::cat({index_tensor,torch::tensor(Multidimension_Index[Dim],torch::kLong).unsqueeze(0)});
    }
    auto Sparse_Tensor_Size_Tensor = torch::tensor(Sparse_Tensor_Size,torch::kLong);
    auto sparse_tensor = torch::sparse_coo_tensor(index_tensor,value_tensor,Sparse_Tensor_Size_Tensor.sizes());
    return sparse_tensor;
}

GridGraphView<bool> GlobalRouter::generate_2d_estimated_congestion_map_by_steinertree(int Gcell_Num_X,int Gcell_Num_Y,std::vector<TorchEdge>& two_Pin_Net_Location){

    std::vector<std::vector<std::vector<int>>> estimated_demand_map_by_steinertree(2,std::vector<std::vector<int>>(Gcell_Num_X,std::vector<int>(Gcell_Num_Y,0)));
    for(auto single_twopinnet:two_Pin_Net_Location){
        for(int x_location=single_twopinnet.pin1.x;x_location<=single_twopinnet.pin2.x;x_location++){
            estimated_demand_map_by_steinertree[0][x_location][single_twopinnet.pin1.y]+=1;
            estimated_demand_map_by_steinertree[0][x_location][single_twopinnet.pin2.y]+=1;
        }
        for(int y_location=single_twopinnet.pin1.y;y_location<=single_twopinnet.pin2.y;y_location++){
            estimated_demand_map_by_steinertree[1][single_twopinnet.pin1.x][y_location]+=1;
            estimated_demand_map_by_steinertree[1][single_twopinnet.pin2.x][y_location]+=1;
        }
    }
    GridGraphView<bool> estimated_congestion_map_by_steinertree;
    estimated_congestion_map_by_steinertree.assign(2,std::vector<std::vector<bool>>(Gcell_Num_X,std::vector<bool>(Gcell_Num_Y,false)));
    //std::vector<std::vector<std::vector<bool>>> estimated_congestion_map_by_steinertree(2,std::vector<std::vector<bool>>(Gcell_Num_X,std::vector<bool>(Gcell_Num_Y,false)));
    int threshold_for_congestion=5;
    int congestion_gcell_count=0;
    for(int x_location=0;x_location<Gcell_Num_X;x_location++){
        for(int y_location=0;y_location<Gcell_Num_Y;y_location++){
            if(estimated_demand_map_by_steinertree[0][x_location][y_location]>threshold_for_congestion) {
                estimated_congestion_map_by_steinertree[0][x_location][y_location]=true;
                congestion_gcell_count++;
            }
            if(estimated_demand_map_by_steinertree[1][x_location][y_location]>threshold_for_congestion) {
                estimated_congestion_map_by_steinertree[1][x_location][y_location]=true;
                congestion_gcell_count++;
            }
            
        }
        
    }
    std::cout<<"congestion_gcell_count:"<<congestion_gcell_count<<std::endl;
    std::cout<<"congestion_gcell_proportion_steinertree:"<<double(congestion_gcell_count)/(2*Gcell_Num_X*Gcell_Num_Y)<<std::endl;
    //std::cout<<estimated_congestion_map_by_steinertree<<std::endl;
    
    return estimated_congestion_map_by_steinertree;
}

GridGraphView<bool> GlobalRouter::generate_2d_estimated_congestion_map_by_RUDY(float p_horizonal,float p_vertical,int Gcell_Num_X,int Gcell_Num_Y,vector<GRNet>& nets){
    std::vector<std::vector<std::vector<float>>> estimated_demand_map_by_RUDY(2,std::vector<std::vector<float>>(Gcell_Num_X,std::vector<float>(Gcell_Num_Y,0)));
    for(const auto& net :nets){
        float dn_horizonal=net.getBoundingBox().width()*p_horizonal/net.getBoundingBox().area();
        float dn_vertical=net.getBoundingBox().height()*p_vertical/net.getBoundingBox().area();
        
        for(auto x_location = net.getBoundingBox().lx(); x_location <= net.getBoundingBox().hx(); x_location++){
            for(auto y_location = net.getBoundingBox().ly(); y_location <= net.getBoundingBox().hy(); y_location++){
                estimated_demand_map_by_RUDY[0][x_location][y_location]+=dn_horizonal;
                estimated_demand_map_by_RUDY[1][x_location][y_location]+=dn_vertical;
            }
        }
    }
    GridGraphView<bool> estimated_congestion_map_by_RUDY;
    estimated_congestion_map_by_RUDY.assign(2,std::vector<std::vector<bool>>(Gcell_Num_X,std::vector<bool>(Gcell_Num_Y,false)));
    float threshold_for_congestion=5;
    int congestion_gcell_count=0;
    for(int x_location=0;x_location<Gcell_Num_X;x_location++){
        for(int y_location=0;y_location<Gcell_Num_Y;y_location++){
            if(estimated_demand_map_by_RUDY[0][x_location][y_location]>threshold_for_congestion) {
                estimated_congestion_map_by_RUDY[0][x_location][y_location]=true;
                congestion_gcell_count++;
            }
            if(estimated_demand_map_by_RUDY[1][x_location][y_location]>threshold_for_congestion) {
                estimated_congestion_map_by_RUDY[1][x_location][y_location]=true;
                congestion_gcell_count++;
            }
            
        }
        
    }
    std::cout<<"congestion_gcell_count_RUDY:"<<congestion_gcell_count<<std::endl;
    std::cout<<"congestion_gcell_proportion_RUDY:"<<double(congestion_gcell_count)/(2*Gcell_Num_X*Gcell_Num_Y)<<std::endl;
    //std::cout<<estimated_congestion_map_by_RUDY<<std::endl;
    return estimated_congestion_map_by_RUDY;
    
}

bool GlobalRouter::check_Tree_Overflow_2D_estimated_congestion_map(std::vector<TorchEdge>& two_Pin_Net_Location,GridGraphView<bool>& Estimated_Congestion_Map){
    for(auto single_twopinnet:two_Pin_Net_Location){
        bool horizontally_overflow_y1=false;
        bool horizontally_overflow_y2=false;
        bool vertically_overflow_x2=false;
        bool vertically_overflow_x1=false;
        for(int x_location=single_twopinnet.pin1.x;x_location<=single_twopinnet.pin2.x;x_location++){
            //std::cout<<"进去了"<<std::endl;
            //std::cout<<Estimated_Congestion_Map[0][0][0]<<std::endl;
            if(Estimated_Congestion_Map[0][x_location][single_twopinnet.pin1.y]==true) horizontally_overflow_y1=true;
            if(Estimated_Congestion_Map[0][x_location][single_twopinnet.pin2.y]==true) horizontally_overflow_y2=true;
        }
        for(int y_location=single_twopinnet.pin1.y;y_location<=single_twopinnet.pin2.y;y_location++){
            //std::cout<<"进去了"<<std::endl;
            if(Estimated_Congestion_Map[1][single_twopinnet.pin2.x][y_location]==true) vertically_overflow_x2=true;
            if(Estimated_Congestion_Map[1][single_twopinnet.pin1.x][y_location]==true) vertically_overflow_x1=true;
        }
        if((((horizontally_overflow_y1==true) || (vertically_overflow_x2==true)) && ((horizontally_overflow_y2==true) || (vertically_overflow_x1==true)))==true) return true;
    }
    return false;//如果这个tree里的所有分支扫了一遍，发现没有overflow的分支，那说明这个tree没有overflow
}



// torch::Tensor GlobalRouter::generate_2d_estimated_congestion_map_by_steinertree(int Gcell_Num_X,int Gcell_Num_Y,std::vector<TorchEdge>& two_Pin_Net_Location){
//     std::vector<std::vector<std::vector<int>>> estimated_demand_map_by_steinertree(2,std::vector<std::vector<int>>(Gcell_Num_X,std::vector<int>(Gcell_Num_Y,0)));
//     // torch::Tensor estimated_demand_map_by_steinertree = torch::zeros({2,Gcell_Num_X,Gcell_Num_Y});
//     for(auto single_twopinnet:two_Pin_Net_Location){
//         for(int x_location=single_twopinnet.pin1.x;x_location<single_twopinnet.pin2.x;x_location++){
//             estimated_demand_map_by_steinertree[0][x_location][single_twopinnet.pin1.y]+=1/2;
//             estimated_demand_map_by_steinertree[0][x_location][single_twopinnet.pin2.y]+=1/2;
//         }
//         for(int y_location=single_twopinnet.pin1.y;y_location<single_twopinnet.pin2.y;y_location++){
//             estimated_demand_map_by_steinertree[1][single_twopinnet.pin1.x][y_location]+=1/2;
//             estimated_demand_map_by_steinertree[1][single_twopinnet.pin2.x][y_location]+=1/2;
//         }
//     }
//     torch::Tensor estimated_demand_map_by_steinertree_Tensor = torch::tensor(estimated_demand_map_by_steinertree,torch::kLong);//多维的vector似乎不能直接转为tensor
//     return estimated_demand_map_by_steinertree_Tensor;
// }


// #include <torch/script.h>
// #include <torch/torch.h>
// #include <c10/cuda/CUDAStream.h>
// #include 
// #include <ATen/cuda/CUDAEvent.h>
 
// #include <iostream>
// #include <memory>
// #include <string>
 
// #include <cuda_runtime_api.h>
// using namespace std;
 
// static void print_cuda_use( )
// {
//     size_t free_byte;
//     size_t total_byte;
 
//     cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
 
//     if (cudaSuccess != cuda_status) {
//         printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
//         exit(1);
//     }
 
//     double free_db = (double)free_byte;
//     double total_db = (double)total_byte;
//     double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;
//     std::cout << "Now used GPU memory " << used_db_1 << "  MB\n";
// }



