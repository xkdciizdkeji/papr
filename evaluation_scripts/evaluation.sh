data_list=("ariane133_51" "ariane133_68" "bsg_chip" "mempool_tile" "nvdla")
input_path="/home/luxu/cugr/benchmark/Simple_inputs"
output_path="/home/luxu/cugr/gr_gpu_1/run0131"
for data in "${data_list[@]}"
do
    # run the whole framework
    echo "data: $data"
    ./evaluator $input_path/$data.cap $input_path/$data.net $output_path/$data-3.out
done

