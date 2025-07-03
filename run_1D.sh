#!/bin/bash

## -----------------------------------------------------------------
# Backward Diffusion
# acquire the current directory current_dir
current_dir=$(pwd)
# print the directory for verification
new_dir="$current_dir/BackwardDiffusion/1D_meta_learning"
cd "$new_dir"
echo "current_dir: $PWD"

> record_time_simple.txt
> record_time_complex.txt

# Get total number of CPU threads available
TOTAL_THREADS=$(nproc)
# Calculate half of available threads for each group
HALF_THREADS=$((TOTAL_THREADS / 2))

echo "System total threads: $TOTAL_THREADS"
echo "Threads allocated per group: $HALF_THREADS"

# Group 1 (simple environment tasks)
(
    echo "Starting simple environment tasks"
    python generate_meta_data.py --env "simple"
    python meta_learn_mean.py --env "simple"
    python generate_meta_data.py --test_true --env "simple"
    python meta_learn_FNO.py --env "simple"
    python MAPSimpleCompare.py
    python var_simple_analysis.py
    echo "Completed simple environment tasks"
) &
PID1=$!

# Group 2 (complex environment tasks)
(
    echo "Starting complex environment tasks"
    python generate_meta_data.py --env "complex"
    python meta_learn_mean.py --env "complex"
    python generate_meta_data.py --test_true --env "complex"
    python meta_learn_FNO.py --env "complex"
    python MAPComplexCompare.py
    python var_complex_analysis.py
    echo "Completed complex environment tasks"
) &
PID2=$!

# Optionally set CPU affinity for each group
if command -v taskset &> /dev/null; then
    taskset -cp 0-$((HALF_THREADS-1)) $PID1
    taskset -cp $HALF_THREADS-$((TOTAL_THREADS-1)) $PID2
fi
# Wait for both task groups to complete
wait

taskset -c 0-10 python meta_learn_mean_test_L.py --env "simple" --Ls 1 &
taskset -c 11-20 python meta_learn_mean_test_L.py --env "simple" --Ls 5 &
taskset -c 21-30 python meta_learn_mean_test_L.py --env "simple" --Ls 10 &
taskset -c 31-40 python meta_learn_mean_test_L.py --env "simple" --Ls 15 &
taskset -c 41-50 python meta_learn_mean_test_L.py --env "simple" --Ls 20 &
wait

taskset -c 0-10 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 5 &
taskset -c 11-20 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 10 &
taskset -c 21-30 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 15 &
taskset -c 31-40 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 20 &
taskset -c 41-50 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 25 &
taskset -c 51-60 python meta_learn_FNO_test_hidden_dim.py --env "complex" --hidden_dims 30 &
wait

python analysis_maxiter_L.py
python analysis_Dh.py

echo "All tasks completed successfully"

