#!/bin/bash

## Job file for a specific config

## Don't change any of these
#SBATCH --job-name=a3_config_2
#SBATCH --partition=all
#SBATCH --constraint="xs-4114*2"
#SBATCH --cpus-per-task 2
#SBATCH --nodes 2
#SBATCH --ntasks 20

## You can change any of the settings below
## Time limit for the job, but change this as necessary.
#SBATCH --time=00:05:00
## Just useful logfile names
#SBATCH --output=logs/a3_%j.slurmlog
#SBATCH --error=logs/a3_%j.slurmlog

# Run the common job file and pass all arguments to it
./job.sh $@