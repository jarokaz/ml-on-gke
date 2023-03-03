#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -o errexit
set -o nounset

trap 'exit_handler $? $LINENO'  ERR  

exit_handler() {
    echo "Error $1 occured in line $2"

    exit 1
    
}

usage()
{
    echo "entrypoint.sh --num_processes <N> <train.py parameters>"
    exit 2
}

get_coordinator_ip_address() {
    
    coordinator_fqdn="${JOB_NAME}-0.${SUB_DOMAIN}"
    coordinator_ip_address=''
    echo "Resolving coordinator IP addres: $coordinator_fqdn"

    counter=5 
    while [ $counter -gt 0 ]; do
        if [ -z "$coordinator_ip_address" ]; then
            sleep 1
        fi
        coordinator_ip_address=$(host $coordinator_fqdn  | awk '/has address/ { print $4 }')
        counter=$(( $counter -1 ))
    done

    if [ -z "$coordinator_ip_address" ]; then
        echo "Failed to resolve Coordinator IP address"
        return 1
    fi 
    echo "Coordinator IP address: $coordinator_ip_address"

}

if [ -z ${JOB_COMPLETION_INDEX+x} ]; then
    echo "JOB_COMPLETION_INDEX not set. Exiting"
    exit_handler 1 "$LINENO" 
fi

long_flags=run_mode:,model_dir:,gin_file:,gin_bindings:,gin_search_paths:,tfds_data_dir:,seqio_additional_cache_dirs:,process_count:
parsed_argumenta=$(getopt  --options '' --long "$long_flags" -- "$@")
valid_argumentS=$?

if [ "$valid_argumentS" != "0" ]; then
    usage
fi

command_line_parameters=()

eval set -- "$parsed_argumenta"
while [ : ]; do
    case "$1" in
    --run_mode)                    run_mode=$2                                        ; shift 2 ;;
    --model_dir)                   model_dir=$2                                       ; shift 2 ;;
    --tfds_data_dir)               command_line_parameters+=("--tfds_data_dir=$2")    ; shift 2 ;;
    --gin_file)                    command_line_parameters+=("--gin_file=$2")         ; shift 2 ;;
    --gin_bindings)                command_line_parameters+=("--gin_bindings=$2")     ; shift 2 ;;
    --gin_search_paths)            command_line_parameters+=("--gin_search_paths=$2") ; shift 2 ;;
    --seqio_additional_cache_dirs) command_line_parameters+=("--gin_bindings=$2")     ; shift 2 ;;
    --) shift; break ;;
    *) shift; usage ;;
    esac
done

time_stamp=$(date +"%m%d%Y-%H%M%S")
job_dir="${model_dir}/${time_stamp}"
command_line_parameters+=("--gin.MODEL_DIR=\"$job_dir\"")

echo $run_mode

if [ "$PROCESS_COUNT" -gt 1 ]; then
    if [ "$run_mode" != train ]; then
        echo "Multi-host jobs are only supported for the Train run mode"
        exit_handler 1 "$LINENO"
    fi
    get_coordinator_ip_address
    command_line_parameters+=( "--process_count=$PROCESS_COUNT" "--multiprocess_gpu" "--process_index=$JOB_COMPLETION_INDEX" "--coordinator_address=$coordinator_ip_address:$COORDINATOR_PORT")
fi


T5X_DIR=/t5x/

if [ "$run_mode" != train ]; then
    command_line_parameters+=("--run_mode=$run_mode")
    script="${T5X_DIR}t5x/main.py"
else
    script="${T5X_DIR}t5x/train.py"
fi

echo "Starting the job with: $script $command_line_parameters"

echo "******** Saving job artifacts to:  $job_dir"

python3 "$script" "${command_line_parameters[@]}" 

echo "Job artifacts can be found in: $job_dir"

