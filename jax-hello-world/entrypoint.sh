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
    
    COORDINATOR_FQDN="${JOB_NAME}-0.${SUB_DOMAIN}"
    COORDINATOR_IP_ADDRESS=''
    echo "Resolving coordinator IP addres: $COORDINATOR_FQDN"

    counter=5 
    while [ $counter -gt 0 ]; do
        if [ -z "$COORDINATOR_IP_ADDRESS" ]; then
            sleep 1
        fi
        COORDINATOR_IP_ADDRESS=$(host $COORDINATOR_FQDN  | awk '/has address/ { print $4 }')
        counter=$(( $counter -1 ))
    done

    if [ -z "$COORDINATOR_IP_ADDRESS" ]; then
        echo "Failed to resolve Coordinator IP address"
        return 1
    fi 
    echo "Coordinator IP address: $COORDINATOR_IP_ADDRESS"

}

if [ -z ${JOB_COMPLETION_INDEX+x} ]; then
    echo "JOB_COMPLETION_INDEX not set. Exiting"
    exit_handler 1 "$LINENO" 
fi

LONG_FLAGS=gin_file:,gin_bindings:,gin_search_paths:,tfds_data_dir:,seqio_additional_cache_dirs:,process_count:
PARSED_ARGUMENTS=$(getopt  --options '' --long "$LONG_FLAGS" -- "$@")
VALID_ARGUMENTS=$?

if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

COMMAND_LINE_PARAMETERS=()

eval set -- "$PARSED_ARGUMENTS"
while [ : ]; do
    case "$1" in
    --tfds_data_dir)               COMMAND_LINE_PARAMETERS+=("--tfds_data_dir=$2")    ; shift 2 ;;
    --gin_file)                    COMMAND_LINE_PARAMETERS+=("--gin_file=$2")         ; shift 2 ;;
    --gin_bindings)                COMMAND_LINE_PARAMETERS+=("--gin_bindings=$2")     ; shift 2 ;;
    --gin_search_paths)            COMMAND_LINE_PARAMETERS+=("--gin_search_paths=$2") ; shift 2 ;;
    --seqio_additional_cache_dirs) COMMAND_LINE_PARAMETERS+=("--gin_bindings=$2")     ; shift 2 ;;
    --) shift; break ;;
    *) shift; usage ;;
    esac
done


if [ "$PROCESS_COUNT" -gt 1 ]; then
    get_coordinator_ip_address
    COMMAND_LINE_PARAMETERS+=( "--process_count=$PROCESS_COUNT" "--multiprocess_gpu" "--process_index=$JOB_COMPLETION_INDEX" "--coordinator_address=$COORDINATOR_IP_ADDRESS")
fi


echo "Invoking train.py ${COMMAND_LINE_PARAMETERS[@]}"

python3 /scripts/train.py "${COMMAND_LINE_PARAMETERS[@]}"

