#!/bin/bash
# SBATCH --account=bbth-delta-cpu
# SBATCH --partition=cpu
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=8 # allows for threading
# SBATCH --mem=200G


# SBATCH -t 03:00:00    # 30 minute wall clock time

# module load modtree/cpu
# module load gcc anaconda3_cpu

echo "Job Deployed, Start Running..."

python_script="point_cloud.py"

for current_case in "./data"/*; do
    case="$(basename "$current_case")"
    echo "$case"
    if [ "$case" != "test04102023" ] && [ "$case" != "test04242023" ]; then
        continue
    fi
    for current_scenario in "$current_case"/*; do
        scenario="$(basename "$current_scenario")"
        if [[ $scenario == "constants" ]]; then
            continue
        fi

        echo "Current Processing Data: $scenario under $case"
        # current_nframes=$(srun python "./check_data_length.py" "$case" "$scenario" | tr -d '\n')
        current_nframes=$(python "./check_data_length.py" "$case" "$scenario" | tr -d '\n')
        for ((frame = 0; frame < current_nframes; frame++)); do
            if [ -f "$current_scenario/frames_point_cloud/Point_Cloud:frame=$current_nframes.npy" ]; then
                echo "Frame $frame has been processed, proceed to next frame..."
            else
                # srun python "$python_script" "$case" "$scenario" --frame="$frame"
                python "$python_script" "$case" "$scenario" --frame="$frame"
                echo "Frame $frame processing done!"
            fi
        done
    done
done