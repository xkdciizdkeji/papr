#rm -rf build/
#cmake -B build -DCMAKE_BUILD_TYPE=Release
#cmake --build build

cd build/
./route -cap ../Simple_inputs/ariane133_51.cap -net ../Simple_inputs/ariane133_51.net -output ../Simple_inputs/ariane133_51.guide
#./route -cap ../Simple_inputs/ariane133_68.cap -net ../Simple_inputs/ariane133_68.net -output ../Simple_inputs/ariane133_68.guide
#./route -cap ../Simple_inputs/bsg_chip.cap -net ../Simple_inputs/bsg_chip.net -output ../Simple_inputs/bsg_chip.guide
#./route -cap ../Simple_inputs/mempool_tile.cap -net ../Simple_inputs/mempool_tile.net -output ../Simple_inputs/mempool_tile.guide
#./route -cap ../Simple_inputs/nvdla.cap -net ../Simple_inputs/nvdla.net -output ../Simple_inputs/nvdla.guide