#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <benchmark_folder>"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "Benchmark folder $1 does not exist"
    exit 1
fi

benchmark_folder=$(basename $1)
echo "Running benchmark for folder at $benchmark_folder"
mkdir -p "$benchmark_folder/output"
declare -a job_id_maps

echo "Running gen_configs for benchmark"
python3 $benchmark_folder/gen_configs.py

for config in "$benchmark_folder"/output/colext_configs/*; do
    echo "Launching job based on config at: $config"
    colext_launch_job -c $config
    # colext_launch_job -c $config > colext_launch_job.output 2>&1
    echo "Job finished"

    # colext_launch_job blocks until job is finished
    # Once the job is finished, pods will still be available to obtain the job_id
    job_id=$(mk get pod fl-server  -o=jsonpath='{.metadata.labels.colext-job-id}')
    if [ -z $job_id ]; then
        echo "ERROR: Could not find job id in fl-server pod labels. Stopping benchmark."
        exit 1
    fi
    echo "Finished experiment with job id = $job_id"
    config_id=$(basename "$config" .yaml | cut -d_ -f2-)
    job_id_maps+=("$job_id=$config_id")
done

echo
echo "Finished all experiments"
# Print all job ids as comma separated list and save it in a file for future use
output_file="$benchmark_folder/output/output_job_id_maps_`date +%Y%m%d%H%M%S`.txt"
printed_map=$(printf '%s\n' "${job_id_maps[@]}")
printf "Job IDs: [\n%s\n]\n" "$printed_map"
echo "$printed_map" > $output_file
