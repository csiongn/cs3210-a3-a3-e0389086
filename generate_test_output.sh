#!/bin/bash

## Generates the ground truth .out file for a given .in test file and arguments

## EXAMPLE: ./generate_test_output.sh 2 1 1 0.00 tests/tinkywinky.in
## This creates the output file tests/tinkywinky_2_1_1_0.00.out

echo "Executing task with parameters $@"
if [ $# -ne 5 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3> <arg4> <arg5>"
    exit 1
fi

arg1=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5
output_file="${arg5%.*}_${arg1}_${arg2}_${arg3}_${arg4}.out"
echo "Putting output in file $output_file"

set -x

# Don't evaluate (it'll just be evaluating itself!)
# Make it sequential for the ground truth
## Use xw to run since it's underutilized and has fastest single core performance
## Uses a single node and task, since we don't need the entire resources of the full config 
##  (we just use 1 rank to do the work)
EVALUATE=0 SEQ=1 sbatch --nodes 1 --ntasks 1 --time 00:05:00 --partition xw-2245 --constraint xw-2245 -o $output_file -e $output_file config1.sh $@