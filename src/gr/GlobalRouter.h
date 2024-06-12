#pragma once
#include "global.h"
#include "obj/Design.h"
#include "GridGraph.h"
#include "GRNet.h"

//Modified by IrisLin
#include "Torchroute.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
//#include <c10/cuda/CUDAGuard.h>

class GlobalRouter {
public:
    GlobalRouter(const Design& design, const Parameters& params);
    void route();
    void write(std::string guide_file = "");

    //Modified by IrisLin
    c10::DeviceType* get_optim_device(c10::DeviceType *dev);
    c10::Device* get_optim_device_fix(c10::Device dev);
    // void create_masks(int gcell_num_x, int gcell_num_y, int pattern_num, std::vector<TorchEdge> edges, torch::Tensor *mask_h, torch::Tensor *mask_v, c10::DeviceType device);
    // void create_masks_fixed(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y, std::vector<TorchEdge> edges,torch::Tensor *mask, c10::DeviceType device);
    void create_masks_fixed_agian(int Layer_Num,int Gcell_Num_X, int Gcell_Num_Y, std::vector<TorchEdge> Two_Pin_net_vector,torch::Tensor *mask, c10::Device device);//针对四维的P矩阵设置的mask
    std::vector<int> compress_multidimensional_index_to_one_dimensional_index(std::vector<int>& multidimension_maxvalue,std::vector<std::vector<int>>& multidimension_index);
    std::vector<std::array<int, 2>> linkBetweenSelectedPatternLayerAndLPatternIndex(int layer_num);
    torch::Tensor generate_sparse_tensor_based_on_multidimensional_index(std::vector<std::vector<int>>& Multidimension_Index,int Value_In_Sparse_Tensor,std::vector<int> Sparse_Tensor_Size);
    GridGraphView<bool> generate_2d_estimated_congestion_map_by_steinertree(int Gcell_Num_X,int Gcell_Num_Y,std::vector<TorchEdge>& two_Pin_Net_Location);
    GridGraphView<bool> generate_2d_estimated_congestion_map_by_RUDY(float p_horizonal,float p_vertical,int Gcell_Num_X,int Gcell_Num_Y,vector<GRNet>& nets);
    //torch::Tensor generate_2d_estimated_congestion_map_by_steinertree(int Gcell_Num_X,int Gcell_Num_Y,std::vector<TorchEdge>& two_Pin_Net_Location);
    bool check_Tree_Overflow_2D_estimated_congestion_map(std::vector<TorchEdge>& two_Pin_Net_Location,GridGraphView<bool>& Estimated_Congestion_Map);
    // torch::Tensor create_Parraymask(std::vector<std::vector<std::vector<TorchEdge>>> Two_Pin_net_vector);
    // void create_Ptreemask();
    
private:
    const Parameters& parameters;
    GridGraph gridGraph;
    //GridGraph gridGraph0;//我想作为pretend过程中使用的gridgraph
    vector<GRNet> nets;
    
    int areaOfPinPatches;
    int areaOfWirePatches;
    
    void sortNetIndices(vector<int>& netIndices) const;
    void getGuides(const GRNet& net, vector<std::pair<int, utils::BoxT<int>>>& guides) ;
    void printStatistics() const;
};


//Modified by IrisLin
struct Torchroute : torch::nn::Module
{
	Torchroute(int net_num,int max_tree_num_in_one_single_net, int two_pin_net_num, int pattern_num)
	{
		//P_pattern = register_parameter("P_pattern", torch::ones({two_pin_net_num,max_tree_num_in_one_single_net,max_two_pin_net_num_in_one_single_tree, pattern_num}));
        P_pattern = register_parameter("P_pattern", torch::ones({two_pin_net_num, pattern_num}));
        P_tree=register_parameter("P_tree", torch::ones({net_num,max_tree_num_in_one_single_net }));
	}
	torch::Tensor forward(int en, int pn, torch::Tensor maskh, torch::Tensor maskv, torch::Tensor capacity, int xsize, int ysize, c10::DeviceType dev)
	{
		auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).reshape({en * pn, 1});
        //auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).reshape({en * pn, 1})*torch::nn::functional::gumbel_softmax(P_tree, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).unsqueeze(P_pattern.size(1)).reshape({en * pn, 1});
        //auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(3)).reshape({P_pattern.size(0) * P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1})*torch::nn::functional::gumbel_softmax(P_tree, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).unsqueeze(P_pattern.size(2)).unsqueeze(P_pattern.size(3)).reshape({P_pattern.size(0) * P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1});
        auto size=maskh.sizes();
        std::cout<<size<<" "<<size.size()<<std::endl;
        // auto p_reshape = torch::nn::functional::gumbel_softmax(p, 1).reshape({en * pn, 1});

		// NO LONGER USED
		// auto wiredemand_h = torch::_sparse_mm(mask_h, p);
		// auto wiredemand_v = torch::_sparse_mm(mask_v, p);
		// cout << wiredemand_h.device() << endl;
		// cout << wiredemand_v.device() << endl;
		// auto of_h = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_h, p), Gcell_capacity_), torch::_sparse_mm(mask_h, p))));
		// auto of_v = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_v, p), Gcell_capacity_), torch::_sparse_mm(mask_v, p))));
		// cout << torch::full_like(wiredemand_h, Gcell_capacity_) << endl;
		// cout << torch::sub(torch::full_like(wiredemand_h, Gcell_capacity_), wiredemand_h) << endl;
		// cout << of_h << endl;
		// cout << of_v << endl;
        torch::Tensor demand_h = torch::_sparse_mm(maskh, p_reshape);
        auto demand_h_ = torch::zeros({xsize*ysize, 1}).to(dev);
        auto demand_v_ = torch::zeros({xsize*ysize, 1}).to(dev);
        demand_h_.slice(0, 0, demand_h.size(0)) = demand_h;
        // auto cost_h = torch::relu(torch::neg(torch::sub(torch::full_like(demand_h, 2.0), demand_h)));
        auto cost_h = torch::relu(torch::sub(demand_h_, capacity[0]))*3800;
        torch::Tensor demand_v = torch::_sparse_mm(maskv, p_reshape);
        demand_v_.slice(0, 0, demand_v.size(0)) = demand_v;
        // auto cost_v = torch::relu(torch::neg(torch::sub(torch::full_like(demand_v, 2.0), demand_v)));
        auto cost_v = torch::relu(torch::sub(demand_v_, capacity[1]))*3800;
        auto total_loss = torch::add(torch::sum(cost_h), torch::sum(cost_v));

		// auto total_loss = torch::sum(torch::add(torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskh, p_reshape), 10), torch::_sparse_mm(maskh, p_reshape)))), torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskv, p_reshape), 10), torch::_sparse_mm(maskv, p_reshape))))));
		// cout << "total_loss: " << total_loss << endl;
		// cout << total_loss.device() << endl;
		return total_loss;
	}
    torch::Tensor forwardfixed(int en, int pn, torch::Tensor *mask, torch::Tensor capacity, int layersize,int xsize, int ysize, c10::DeviceType dev)
	{
		//auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).reshape({en * pn, 1});
        auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(3)).reshape({P_pattern.size(0) * P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1})*torch::nn::functional::gumbel_softmax(P_tree, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).unsqueeze(P_pattern.size(2)).unsqueeze(P_pattern.size(3)).reshape({P_pattern.size(0) * P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1});
        
        for (int64_t dim : p_reshape.sizes()) std::cout<<dim<<" ";
        
        capacity=capacity.reshape({layersize*xsize*ysize, 1}).to(dev);
        std::cout<<"lastok"<<std::endl;
        auto size=mask->sizes();
        std::cout<<size<<" "<<size.size()<<std::endl;
        
        // auto p_reshape = torch::nn::functional::gumbel_softmax(p, 1).reshape({en * pn, 1});

		// NO LONGER USED
		// auto wiredemand_h = torch::_sparse_mm(mask_h, p);
		// auto wiredemand_v = torch::_sparse_mm(mask_v, p);
		// cout << wiredemand_h.device() << endl;
		// cout << wiredemand_v.device() << endl;
		// auto of_h = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_h, p), Gcell_capacity_), torch::_sparse_mm(mask_h, p))));
		// auto of_v = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_v, p), Gcell_capacity_), torch::_sparse_mm(mask_v, p))));
		// cout << torch::full_like(wiredemand_h, Gcell_capacity_) << endl;
		// cout << torch::sub(torch::full_like(wiredemand_h, Gcell_capacity_), wiredemand_h) << endl;
		// cout << of_h << endl;
		// cout << of_v << endl;

        //auto all_gcell_demand=torch::_sparse_mm(mask, p_reshape)
        auto all_gcell_demand=torch::_sparse_mm(*mask, p_reshape);
        //torch::Tensor all_gcell_demand = torch::_sparse_mm(mask, p_reshape);
        std::cout<<"thisisok"<<std::endl;
        auto all_gcell_demand_ = torch::zeros({layersize*xsize*ysize, 1}).to(dev);
        std::cout<<"ok"<<std::endl;
        all_gcell_demand_.slice(0, 0, all_gcell_demand.size(0)) = all_gcell_demand;
        std::cout<<"ok"<<std::endl;
        auto cost = torch::relu(torch::sub(all_gcell_demand_, capacity))*3800;
        std::cout<<"ok"<<std::endl;
        auto total_loss = torch::sum(cost);


        // torch::Tensor demand_h = torch::_sparse_mm(maskh, p_reshape);
        // auto demand_h_ = torch::zeros({xsize*ysize, 1}).to(dev);
        // auto demand_v_ = torch::zeros({xsize*ysize, 1}).to(dev);
        // demand_h_.slice(0, 0, demand_h.size(0)) = demand_h;
        // // auto cost_h = torch::relu(torch::neg(torch::sub(torch::full_like(demand_h, 2.0), demand_h)));
        // auto cost_h = torch::relu(torch::sub(demand_h_, capacity[0]))*3800;
        // torch::Tensor demand_v = torch::_sparse_mm(maskv, p_reshape);
        // demand_v_.slice(0, 0, demand_v.size(0)) = demand_v;
        // // auto cost_v = torch::relu(torch::neg(torch::sub(torch::full_like(demand_v, 2.0), demand_v)));
        // auto cost_v = torch::relu(torch::sub(demand_v_, capacity[1]))*3800;
        // auto total_loss = torch::add(torch::sum(cost_h), torch::sum(cost_v));

		// auto total_loss = torch::sum(torch::add(torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskh, p_reshape), 10), torch::_sparse_mm(maskh, p_reshape)))), torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskv, p_reshape), 10), torch::_sparse_mm(maskv, p_reshape))))));
		// cout << "total_loss: " << total_loss << endl;
		// cout << total_loss.device() << endl;
		return total_loss;
	}
    float initial_t=1;
    torch::Tensor forwardfixed_agian(int net_num,int max_tree_num_in_one_single_net,int All_Two_Pin_Net_Num, int pn, torch::Tensor *mask,torch::Tensor *P_tree_mask_to_P_pattern, torch::Tensor capacity, int layersize,int xsize, int ysize, c10::Device dev)
	{
		//auto p_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).reshape({en * pn, 1});
        initial_t=0.9*initial_t;
        std::cout << "this_epoch_t:"<<initial_t<<std::endl;
        auto P_pattern_reshape = torch::nn::functional::gumbel_softmax(P_pattern, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(initial_t).dim(1)).reshape({All_Two_Pin_Net_Num*pn, 1}).to(dev);
        //auto P_tree_reshape = torch::nn::functional::gumbel_softmax(P_tree, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).unsqueeze(2).unsqueeze(3).expand({P_pattern.size(0),P_pattern.size(1),P_pattern.size(2),P_pattern.size(3)}).reshape({P_pattern.size(0)*P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1}).to(dev);
        //auto P_tree_reshape = torch::nn::functional::gumbel_softmax(P_tree, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(0.1).dim(1)).unsqueeze(2).unsqueeze(3).expand({P_pattern.size(0),P_pattern.size(1),P_pattern.size(2),P_pattern.size(3)}).reshape({P_pattern.size(0)*P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1}).to(dev);
        //auto P_tree_reshape =(torch::_sparse_mm(*P_tree_mask_to_P_pattern,P_tree.reshape({net_num*max_tree_num_in_one_single_net,1}))).reshape({All_Two_Pin_Net_Num,pn});
        auto P_tree_reshape =(torch::_sparse_mm(*P_tree_mask_to_P_pattern,P_tree.reshape({net_num*max_tree_num_in_one_single_net,1}))).repeat({1,pn}).reshape({All_Two_Pin_Net_Num*pn, 1}).to(dev);//P_tree_reshape的尺寸为：{All_Two_Pin_Net_Num*pn, 1}
        auto p_reshape = P_pattern_reshape * P_tree_reshape;
        P_pattern_reshape.reset();
        P_tree_reshape.reset();
        
        
        //auto P_invalid_mask_reshape = P_Invalid_mask->reshape({P_pattern.size(0) * P_pattern.size(1)*P_pattern.size(2)*P_pattern.size(3), 1}).to(dev);
        //for (int64_t dim : p_reshape.sizes()) std::cout<<dim<<" ";
        
        capacity=capacity.reshape({layersize*xsize*ysize, 1}).to(dev);
        
        // auto size=mask->sizes();
        // std::cout<<size<<" "<<size.size()<<std::endl;
        // auto size_mask=P_Invalid_mask->sizes();
        // std::cout<<size_mask<<" "<<size_mask.size()<<std::endl;
        
        // auto p_reshape = torch::nn::functional::gumbel_softmax(p, 1).reshape({en * pn, 1});

		// NO LONGER USED
		// auto wiredemand_h = torch::_sparse_mm(mask_h, p);
		// auto wiredemand_v = torch::_sparse_mm(mask_v, p);
		// cout << wiredemand_h.device() << endl;
		// cout << wiredemand_v.device() << endl;
		// auto of_h = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_h, p), Gcell_capacity_), torch::_sparse_mm(mask_h, p))));
		// auto of_v = torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(mask_v, p), Gcell_capacity_), torch::_sparse_mm(mask_v, p))));
		// cout << torch::full_like(wiredemand_h, Gcell_capacity_) << endl;
		// cout << torch::sub(torch::full_like(wiredemand_h, Gcell_capacity_), wiredemand_h) << endl;
		// cout << of_h << endl;
		// cout << of_v << endl;

        //auto all_gcell_demand=torch::_sparse_mm(mask, p_reshape)
        //system("nvidia-smi");
        //std::cout<<"1"<<std::endl;
        //auto all_gcell_demand=torch::_sparse_mm(*mask, p_reshape);
        auto all_gcell_demand=torch::mm(*mask, p_reshape);
        torch::Tensor temp = torch::rand({all_gcell_demand.sizes()}).to(dev);
        //system("nvidia-smi");
        all_gcell_demand=all_gcell_demand*temp;
        
        std::cout << "this line has been executed" << std::endl;
        //system("nvidia-smi");
        std::cout<<"2"<<std::endl;
        
        
        std::cout<<all_gcell_demand.dtype()<<std::endl;
        std::cout<<all_gcell_demand.sizes()<<std::endl;
        std::cout<<all_gcell_demand.numel()<<std::endl;
        std::cout<<all_gcell_demand.element_size()<<std::endl;
        std::cout<<"all_gcell_demand:"<<all_gcell_demand.element_size()*all_gcell_demand.numel()<<"bits"<<std::endl;
        std::cout<<"p_reshape:"<<p_reshape.element_size()*p_reshape.numel()<<"bits"<<std::endl;
        std::cout<<"mask:"<<mask->element_size()*mask->numel()<<"bits"<<std::endl;
        
        
        


        // //输出all_gcell_demand的尺寸?
        // std::cout<<"all_gcell_demand: "<<all_gcell_demand.sizes()<<std::endl;
        // std::cout<<"all_gcell_demand: "<<all_gcell_demand[0][0]<<all_gcell_demand[10][0]<<all_gcell_demand[100][0]<<all_gcell_demand[1000][0]<<std::endl;
        // //输出capacity的尺寸?
        // std::cout<<"capacity: "<<capacity.sizes()<<std::endl;
        // std::cout<<"capacity: "<<capacity[0][0]<<capacity[10][0]<<capacity[100][0]<<capacity[1000][0]<<std::endl;
        
        //auto all_gcell_demand_invalid=torch::_sparse_mm(*mask, p_reshape*P_invalid_mask_reshape);
        //torch::Tensor all_gcell_demand = torch::_sparse_mm(mask, p_reshape);
        
        //auto cost = torch::relu(torch::sub(all_gcell_demand,capacity));
        auto cost_overflow = torch::relu(torch::sub(all_gcell_demand,capacity))*3800;
        // system("nvidia-smi");
        // std::cout<<"3"<<std::endl;
        // auto cost1 = torch::sub(all_gcell_demand,capacity);
        // //输出cost的尺寸?
        // std::cout<<"cost1: "<<cost1.sizes()<<std::endl;
        // std::cout<<"cost1: "<<cost1[0][0]<<cost1[10][0]<<cost1[100][0]<<cost1[1000][0]<<std::endl;
        // auto cost = torch::relu(cost1);
        // //输出cost的尺寸?
        // std::cout<<"cost: "<<cost.sizes()<<std::endl;
        // std::cout<<"cost: "<<cost[0][0]<<cost[10][0]<<cost[100][0]<<cost[1000][0]<<std::endl;
        //auto cost = all_gcell_demand;
        //auto cost_invalid = torch::relu(torch::sub(all_gcell_demand_invalid,capacity))*1e9;
        //auto total_loss = torch::sum(cost) + torch::sum(cost_invalid);
        auto total_loss = torch::sum(cost_overflow)+torch::sum(all_gcell_demand);
        



        // auto all_gcell_demand_ = torch::zeros({layersize*xsize*ysize, 1}).to(dev);
        // std::cout<<"ok"<<std::endl;
        // all_gcell_demand_.slice(0, 0, all_gcell_demand.size(0)) = all_gcell_demand;
        // std::cout<<"ok"<<std::endl;
        // auto cost = torch::relu(torch::sub(all_gcell_demand_, capacity))*3800;
        // std::cout<<"ok"<<std::endl;
        // auto total_loss = torch::sum(cost);


        // torch::Tensor demand_h = torch::_sparse_mm(maskh, p_reshape);
        // auto demand_h_ = torch::zeros({xsize*ysize, 1}).to(dev);
        // auto demand_v_ = torch::zeros({xsize*ysize, 1}).to(dev);
        // demand_h_.slice(0, 0, demand_h.size(0)) = demand_h;
        // // auto cost_h = torch::relu(torch::neg(torch::sub(torch::full_like(demand_h, 2.0), demand_h)));
        // auto cost_h = torch::relu(torch::sub(demand_h_, capacity[0]))*3800;
        // torch::Tensor demand_v = torch::_sparse_mm(maskv, p_reshape);
        // demand_v_.slice(0, 0, demand_v.size(0)) = demand_v;
        // // auto cost_v = torch::relu(torch::neg(torch::sub(torch::full_like(demand_v, 2.0), demand_v)));
        // auto cost_v = torch::relu(torch::sub(demand_v_, capacity[1]))*3800;
        // auto total_loss = torch::add(torch::sum(cost_h), torch::sum(cost_v));

		// auto total_loss = torch::sum(torch::add(torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskh, p_reshape), 10), torch::_sparse_mm(maskh, p_reshape)))), torch::relu(torch::neg(torch::sub(torch::full_like(torch::_sparse_mm(maskv, p_reshape), 10), torch::_sparse_mm(maskv, p_reshape))))));
		// cout << "total_loss: " << total_loss << endl;
		// cout << total_loss.device() << endl;
		return total_loss;
	}
    int en;
    int pn;
	torch::Tensor P_pattern;
    torch::Tensor P_tree;
    
};