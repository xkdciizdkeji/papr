data_list=("ariane133_68")
input_path="/home/luxu/cugr/benchmark/Simple_inputs"
output_path="/home/luxu/cugr/gr_multi_gpu/run0124"
for data in "${data_list[@]}"
do
    # run the whole framework
    echo "data: $data"
    ./evaluator $input_path/$data.cap $input_path/$data.net $output_path/$data-mt.out
done

