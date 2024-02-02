import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import ast
import time



read_txt_begin_t=time.time()
argv_file_path = './argv.txt'
with open(argv_file_path, 'r') as argv_file:
    argv = ast.literal_eval(argv_file.read())
print(argv)

all_pin_location_array_file_path = './all_pin_location_array.txt'
#file_path = './steinertree/SteinerTree-nvdla.txt'


# 读取文本文件
with open(all_pin_location_array_file_path, 'r') as all_pin_location_array_file:
    # 将文件内容解析为嵌套列表
    

    data_list = ast.literal_eval(all_pin_location_array_file.read())
    
    
#print(data_list)
    
data_array = np.concatenate([np.array(sublist) for sublist in data_list])
data_array = np.insert(data_array,0,0, axis=2)
#print(data_array)

read_txt_end_t=time.time()

#一些全局变量
layer_num = 2
gcell_num_in_x_axis = argv[0]
print(gcell_num_in_x_axis)
gcell_num_in_y_axis = argv[1]
print(gcell_num_in_y_axis)
total_gcell_num=layer_num*gcell_num_in_x_axis*gcell_num_in_y_axis
gcell_edge_lenth_across_x_orientation=1#注意！这是垂直于x轴方向的edge，而不是平行于x方向的edge，实际是平行于y的
gcell_edge_lenth_across_y_orientation=1



gcell_attribute_num=2#每个gcell所具有的属性数量，每个gcell只有两个属性，0：capacity；1：demand；
capacity = 2
demand = 0 #没布线的时候的demand当然为0


# two_pin_net_num = two_pin_net_num_test
two_pin_net_num = len(data_array)
#pin_num = two_pin_net_num * 2
pin_num_in_a_two_pin_net = 2#这个恒定为2，不能改动
link_num_between_two_pins =1 #这个也恒为1，不能改动，两个pin，他们之间只能有1个连接

#我打算如果后面将flute引入以后，将多pin net的布线问题转化为two pin net的布线问题

#print(layer_num%2,layer_num/2)
#算L形走线的可能数
if layer_num%2:#余数为1，则层数为奇数
    L_pattern_num=int(layer_num*(layer_num-1)/2-((int(layer_num/2)+1)*int(layer_num/2)/2+int(layer_num/2)*(int(layer_num/2)-1)/2))*2
else:#层数为偶数
    L_pattern_num=int((layer_num*(layer_num-1)/2-layer_num/2*(layer_num/2-1))*2)
    # [[0 1]
    #  [1 0]
    #  [0 3]
    #  [3 0]
    #  [2 1]
    #  [1 2]
    #  [2 3]
    #  [3 2]]这是以layer_num=4为例打印出的矩阵，通过这种方式建立了pattern（pin1到pin2先走哪层再走哪层）和pattern_index（即为这个矩阵的行下标）之间的关系
#print("L形走线的可能数为:",L_pattern_num)
P_array_elements_num=two_pin_net_num*L_pattern_num


epoch_n=100#优化器优化次数
lr=0.1
Wcong=1#对于惩罚项的权重参数，可以适当设大一些，避免overflow
Wunitwirelength=50
Wvia=1



#定义一个矩阵用来存储pattern到底走哪层和pattern index的信息
#例如：如果layer_num=4，那么这个矩阵可能是这样的：
# [[0 1]
#  [1 0]
#  [0 3]
#  [3 0]
#  [2 1]
#  [1 2]
#  [2 3]
#  [3 2]]
#对于那个P矩阵，它的第一列数据所代表的pattern就是先走0层，后走1层

link_between_selectedpatternlayer_and_L_pattern_index=np.zeros([L_pattern_num,2],int)#2指两列数据，第一列存pin1到pin2先走哪层，第二列存再走哪层
m0=[]#偶数序列
m1=[]#奇数序列
m0=[x for x in range(0,layer_num) if x%2==0]#将偶数层序号存入
m1=[x for x in range(0,layer_num) if x%2==1]
i=0
j=0
k=0
for i in m0:
    for j in m1:
        link_between_selectedpatternlayer_and_L_pattern_index[k]=[i,j]
        k=k+1
        link_between_selectedpatternlayer_and_L_pattern_index[k]=[j,i]
        k=k+1



all_pin_location=data_array

def generate_all_gcell_mask():#产生所有Gcell的mask，最终汇总成{(layer_num*X*Y)*(net_num*L_pattern_num)}的矩阵，这次使用稀疏矩阵
    
    m0=[]
    m1=[]
    
    for net_index in range(0,two_pin_net_num):
        l1=all_pin_location[net_index,0,0]
        x1=all_pin_location[net_index,0,1]
        y1=all_pin_location[net_index,0,2]

        l2=all_pin_location[net_index,1,0]
        x2=all_pin_location[net_index,1,1]
        y2=all_pin_location[net_index,1,2]
        
        #对于这个two pin net，依据两个pin的位置，对走线路径上的gcell的对应的mask进行置1
        #先根据link_between_pattern_and_pattern_index矩阵，针对每个pattern，就可以确定在All_Gcell_Mask置1的位置在哪里
        #例如：加入现在link_between_pattern_and_pattern_index如下：
        #   # [[0 1]
            #  [1 0]
            #  [0 3]
            #  [3 0]
            #  [2 1]
            #  [1 2]
            #  [2 3]
            #  [3 2]]
        
        #根据第一行的[0,1]，那么第一步走0层，第二步走1层。同时，这个布线方法对应了P_array里的第0列元素
        #对于路径上的gcell，操作All_Gcell_Mask[l,x,y,net_index,pattern_index]=1

        for pattern_index in range(0,L_pattern_num):
            first_step_layerindex=link_between_selectedpatternlayer_and_L_pattern_index[pattern_index,0]
            second_step_layerindex=link_between_selectedpatternlayer_and_L_pattern_index[pattern_index,1]
            

            #第一段原地穿孔
            for k in range(l1,first_step_layerindex):
                
                m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+y1)
                m1.append(net_index*L_pattern_num+pattern_index)
                
            for k in range(first_step_layerindex,l1):
                
                m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+y1)
                m1.append(net_index*L_pattern_num+pattern_index)
                
            
            #第二段，由first_step_layer_index决定是先横着走还是先竖着走
            if first_step_layerindex%2==0:#第一步是偶数（第一步在偶数层布线），说明要横向走，那么pattern的拐点在（x2，y1）
                
                for k in range(x1,x2+1):#如果先横向走线，走线第二段  #加一是因为走线走的是当前这个gcell右侧的那个edge
                    
                    m0.append(first_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+k*gcell_num_in_y_axis+y1)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(x2,x1+1):
                    
                    m0.append(first_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+k*gcell_num_in_y_axis+y1)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                
                #走线第三段拐角处穿孔
                for k in range(first_step_layerindex,second_step_layerindex):
                    
                    m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+y1)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(second_step_layerindex,first_step_layerindex):
                    
                    m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+y1)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(y1,y2+1):#如果先横向走线，走线第四段
                    
                    m0.append(second_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+k)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(y2,y1+1):
                    
                    m0.append(second_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+k)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                
            else:#如果第一步是在奇数层布线，说明要先竖向走线，pattern的拐点在（x1，y2）
                for k in range(y1,y2+1):#如果先竖向走线，走线第二段
                    
                    m0.append(first_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+k)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(y2,y1+1):
                    
                    m0.append(first_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+k)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    

                    #走线第三段拐角处穿孔
                for k in range(first_step_layerindex,second_step_layerindex):
                    
                    m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+y2)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(second_step_layerindex,first_step_layerindex):
                    
                    m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x1*gcell_num_in_y_axis+y2)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    

                for k in range(x1,x2+1):#如果先竖向走线，走线第四段
                
                    m0.append(second_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+k*gcell_num_in_y_axis+y2)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    
                for k in range(x2,x1+1):
                    
                    m0.append(second_step_layerindex*gcell_num_in_x_axis*gcell_num_in_y_axis+k*gcell_num_in_y_axis+y2)
                    m1.append(net_index*L_pattern_num+pattern_index)
                    

            #第五段原地穿孔
            for k in range(l2,second_step_layerindex):
                
                m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+y2)
                m1.append(net_index*L_pattern_num+pattern_index)
                
            for k in range(second_step_layerindex,l2):
                
                m0.append(k*gcell_num_in_x_axis*gcell_num_in_y_axis+x2*gcell_num_in_y_axis+y2)
                m1.append(net_index*L_pattern_num+pattern_index)
                

    
    index=torch.tensor([m0,m1])


    
    value=torch.ones(len(index[0]))

    All_Gcell_Mask=torch.sparse_coo_tensor(index, value, (layer_num*gcell_num_in_x_axis*gcell_num_in_y_axis,two_pin_net_num*L_pattern_num))
    print(All_Gcell_Mask)
    return All_Gcell_Mask

all_gcell_mask=generate_all_gcell_mask().cuda()




def calculate_loss(P_Array):
    #print(P_Array)
    loss=0.
    global all_wirelength_loss
    global all_gcell_overflow_loss
    all_wirelength_loss=0

    all_gcell_overflow_loss=0

    #result=torch.mul(all_gcell_mask,P_Array)
    #print(P_Array.reshape(two_pin_net_num*L_pattern_num,1))
    all_gcell_demand=torch.spmm(all_gcell_mask,P_Array.reshape(two_pin_net_num*L_pattern_num,1).cuda()).cuda()
    #all_gcell_demand=torch.mul(all_gcell_mask,P_Array).cuda().sum([3,4]).cuda()#将第3，4维度的矩阵sum一下，现在这个矩阵的尺寸是layer_num*X*Y
    #print(all_gcell_demand)
    
    

    
    #all_wirelength_loss=all_gcell_demand.sum().cuda()
    all_wirelength_loss=0
    #all_gcell_overflow_loss=all_gcell_demand.sub(capacity).clamp(0).mul(Wcong).sum().cuda()#先将所有gcell的demand减2，然后再将小于0的部分钳去，剩下的就是overflow的部分，然后乘个系数，然后sum一下就是overflow造成的loss
    all_gcell_overflow_loss=torch.add(torch.full(all_gcell_demand.shape,-capacity).cuda(),all_gcell_demand).clamp(0).mul(Wcong).sum().cuda()
    #all_gcell_overflow_loss=torch.add(torch.full(all_gcell_demand.shape,-2).cuda(),all_gcell_demand).clamp(0).mul(Wcong).sum().cuda()
    loss=all_wirelength_loss+all_gcell_overflow_loss.cuda()
    
    #final_all_gcell_demand=torch.mul(all_gcell_mask,(P_Array == P_Array.max(dim=1,keepdim=True)[0])).sum([3,4])
    # a=final_all_gcell_demand.sum()
    # b=final_all_gcell_demand.sub(capacity).clamp(0).mul(Wcong).sum()
    # loss=a+b
    return loss




class Mtsroute(nn.Module):
    # initializers
    def __init__(self):
        super(Mtsroute, self).__init__()
        #self.changed_P_array = torch.nn.Parameter(data=torch.ones(two_pin_net_num,L_pattern_num), requires_grad=True)
        self.changed_P_array = torch.nn.Parameter(data=torch.rand(two_pin_net_num,L_pattern_num), requires_grad=True)
        #self.changed_P_array = torch.nn.parameter(data=torch.softmax(self.changed_P_array,1))

    # weight_init
    def weight_init(self):
        self.changed_P_array.data.normal_()

    # # forward method
    # def forward(self):
    #     r=calculate_loss(torch.softmax(self.changed_P_array,1)).cuda()
    #     #r=calculate_loss(update_all_gcell_demand(torch.softmax(self.changed_P_array,1),updated_all_gcell_attribute_information))
        

    #     return r
    def forward(self,Tau):
            #r=calculate_loss(torch.softmax(self.changed_P_array,1)).cuda()
            r=calculate_loss(torch.nn.functional.gumbel_softmax(self.changed_P_array,dim=1,tau=Tau,hard=False)).cuda()
            #r=calculate_loss(update_all_gcell_demand(torch.softmax(self.changed_P_array,1),updated_all_gcell_attribute_information))
            

            return r


m=Mtsroute()


m.weight_init()

m.train()


opt = optim.Adam(m.parameters(), lr=lr,betas=(0.5, 0.999))

last_lost=float("inf")#用来配合早停机制的量
tau=1
for epoch in range(epoch_n):
    
    opt.zero_grad()
    
    global loss
    loss=m.forward(tau).cuda()
    tau-=0.01

    #加入早停机制
    if loss>last_lost:
        break
    last_lost=loss


    #all_gcell_attribute_information=updated_all_gcell_attribute_information#把demand信息放回到all_gcell_attribute_information里

    loss.backward()
    

    
    
    
    opt.step()
    print("No.",epoch+1,"time,the loss is ",loss.item())


print("two_pin_net_num:",two_pin_net_num)
print("loss中线长的部分:",all_wirelength_loss)

print("loss中overf惩罚项部分:",all_gcell_overflow_loss)

P_array=torch.softmax(m.changed_P_array.data,1)#给后续显示输出用


print(P_array)


# #将P_array矩阵输出到一个txt文件里
# with open("origin_P_array.txt","w") as p:
#     for i in range(0,two_pin_net_num):
#         for j in range(0,L_pattern_num):
#             #只输出4位有效数字
#             #p.write(str(":.4f".format(P_array[i][j].item())))
#             p.write(str(P_array[i][j]))
#         p.write("\n")



# 找到每行的最大值
choosed_pattern_index = torch.argmax(P_array, dim=1)



print(choosed_pattern_index)

write_txt_begin_time=time.time()

#将choosed_pattern_index矩阵输出到一个txt文件里
with open("choosed_pattern_index_array.txt","w") as f:
    for i in range(0,two_pin_net_num):
        
        f.write(str(choosed_pattern_index[i].item()))
        f.write("\n")

python_read_txt_time=read_txt_end_t-read_txt_begin_t
print("python_script_read_txt_time:",python_read_txt_time)
python_script_runtime=write_txt_begin_time-read_txt_end_t
print("python_script_runtime:",python_script_runtime)
python_write_txt_time=time.time()-write_txt_begin_time
print("python_script_write_txt_time:",python_write_txt_time)
