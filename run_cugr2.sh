#以下三个目录需要自己更改
#项目目录，即这个程序的目录
root_path="/data1/students/fengjx/cu-gr-2/"
# #官方提供的evaluator所在地址
# evaluator_path="/home/fengjx/eda/evaluation_scripts"

input_path="/data1/students/fengjx/eda/ispd24contest/"
#我将def文件和lef文件统统放在了input_path下


#记录输出的log文件存放在了input_path/log下


#!bin/bash

#测哪些case自己选择
#case_array=("ariane133_51")
#case_array=("ariane133_68")
#case_array=("bsg_chip")
#case_array=("mempool_tile")
#case_array=("nvdla")
#case_array=("ariane133_51" "ariane133_68")
#case_array=("ariane133_51" "ariane133_68" "bsg_chip")
case_array=("ariane133_51" "ariane133_68" "bsg_chip"  "mempool_tile" "nvdla")
case_array=("ariane133_51" "ariane133_68" "mempool_tile" "nvdla")
#case_array=("nvdla")
case_array=("bsg_chip")
#case_array=("mempool_group")
#case_array=("mempool_cluster")
#case_array=("bsg_chip" "nvdla" "mempool_cluster")
#case_array=("bsg_chip" "nvdla" "mempool_cluster" "mempool_cluster_large")
#case_array=("mempool_cluster_large")
#case_array=("ariane133_51" "ariane133_68" "bsg_chip" "mempool_group" "mempool_tile" "nvdla")
#case_array=("ariane133_51" "ariane133_68" "bsg_chip" "mempool_tile" "nvdla" "mempool_group" "mempool_cluster" "mempool_cluster_large")
case_array=("ariane133_51" "ariane133_68" "bsg_chip" "mempool_tile" "nvdla" "mempool_group" "cluster")
case_array=("ariane133_51" "ariane133_68" "bsg_chip" "mempool_tile" "nvdla" "mempool_group")
case_array=("cluster")
case_array=("nvdla")
case_array=("ariane133_51")
case_array=("ariane133_51" "ariane133_68" "mempool_tile" "nvdla" "bsg_chip" "mempool_group")
# case_array=("ariane133_51" "ariane133_68" "mempool_tile" "nvdla" "bsg_chip")
# case_array=("mempool_group")
# case_array=("bsg_chip")
#case_array=("cluster")
#case_array=("mempool_group" "mempool_cluster")
#case_array=("mempool_group" "mempool_cluster" "mempool_cluster_large")
#case_array=("mempool_cluster_large")

cd $input_path
mkdir log
mkdir PR_output
cd PR_output
# mkdir cpu
# mkdir gpu
# mkdir cpu+iss
# mkdir gpu+iss
rm -r $input_path/log/*
cd $root_path

declare -A order
order=(
    [origin]="scripts/build.py -o release",
    [cpu]="cmake -B build -DCMAKE_BUILD_TYPE=Release",
    [gpu]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON",
    [cpu+iss]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON",
    [gpu+iss]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON -DENABLE_CUDA=ON",
    [cpu+multithread]="cmake -B build -DCMAKE_BUILD_TYPE=Release",
    [gpu+multithread]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON",
    [cpu+iss+multithread]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON",
    [gpu+iss+multithread]="cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON -DENABLE_CUDA=ON"
)
#environment=("gpu" "gpu+iss")
#environment=("gpu")
environment=("origin")
#environment=("cpu" "gpu" "cpu+iss" "gpu+iss")
#environment=("cpu+multithread" "gpu+multithread" "cpu+iss+multithread" "gpu+iss+multithread")
#environment=("cpu" "gpu" "cpu+iss" "gpu+iss" "cpu+multithread" "gpu+multithread" "cpu+iss+multithread" "gpu+iss+multithread")






for env in "${environment[@]}"
do
    cd $root_path
    rm -rf build/
    rm -rf run/
    echo "cmake order: ${order[$env]}"
    scripts/build.py -o release
    #${order[$env]}
    #cmake --build build
    
    
    
    cd $input_path"PR_output/"
    mkdir $env
    output_path=$input_path"PR_output/"$env
    log_file_path=$input_path"log/"$env".txt"

    
    for case in "${case_array[@]}"
    do
        echo "case_name: $case     env: $env"
        cd $root_path 
        cd run/
        ./route -lef $input_path/Nangate.lef -def $input_path/$case.def -output $output_path/$case.PR_output -thread 1 | tee -a $log_file_path

        # echo "case_name: $case use evaluator"
        # cd $evaluator_path
        # # ????????log??????????“use evaluator test data: ”,??????????python?????????????????
        # ./evaluator $input_path/$case.cap $input_path/$case.net $output_path/$case.PR_output | tee -a temp_file
        # sed "s/^/use evaluator test data: /" temp_file >> $log_file_path
        # rm temp_file 
    done
    
done

