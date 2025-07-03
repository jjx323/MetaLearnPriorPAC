#!/bin/bash

## -----------------------------------------------------------------
# Darcy flow problem

# print the directory for verification
current_dir=$(pwd)
new_dir="$current_dir/SteadyStateDarcyFlow/2D_meta_learn"
cd "$new_dir"
echo "current_dir: $PWD"

# generate data
# python generate_meta_data.py

# learn mean function and mean function with FNO
# touch record_time.txt
# taskset -c 0-10 python meta_learn_mean.py --env "simple" &
# taskset -c 11-20 python meta_learn_mean.py --env "complex" &
# taskset -c 21-30 python meta_learn_mean_FNO.py --env "simple" &
# taskset -c 31-40 python meta_learn_mean_FNO.py --env "complex" &

taskset -c 0-10 python prepare_test.py --env "simple" &
taskset -c 11-20 python prepare_test.py --env "complex" &
wait

taskset -c 0-10 python run_map.py --env "simple" &
taskset -c 11-20 python run_map.py --env "complex" &
wait

taskset -c 0-10 python compare_truth_FNO.py --env "simple" &
taskset -c 11-20 python compare_truth_FNO.py --env "complex" &
wait

taskset -c 0-10 python run_smc_mix.py --env "simple" &
taskset -c 11-20 python run_smc_mix.py --env "complex" &
wait 

taskset -c 0-10 python smc_mix_analysis.py --env "simple" &
taskset -c 11-20 python smc_mix_analysis.py --env "complex" &

python test_map.py
python analysis_test_map.py

