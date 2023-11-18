#!/bin/bash

## Job file for a specific config

## Don't change any of these
#SBATCH --job-name=a3_config_1
#SBATCH --partition=all
#SBATCH --constraint="[i7-7700*1&xs-4114*1]"
#SBATCH --cpus-per-task 2
#SBATCH --nodes 2
#SBATCH --ntasks 14

## You can change any of the settings below
## Time limit for the job, but change this as necessary.
#SBATCH --time=00:05:00
## Just useful logfile names
#SBATCH --output=logs/a3_%j.slurmlog
#SBATCH --error=logs/a3_%j.slurmlog

# Run the common job file and pass all arguments to it
./job.sh $@