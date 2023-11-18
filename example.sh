#!/bin/bash

## Examples of running your code
set -e

# Make here to avoid any conflicts between overlapping makes
make all

## These are the 5 tests x 3 configs that you must achieve your speedup on

## Config1
# sbatch config1.sh 16 1 2 0.10 tests/tinkywinky.in
# sbatch config1.sh 5 3 5 0.50 tests/dipsy.in
# sbatch config1.sh 5 2 2 0.00 tests/lala.in
# sbatch config1.sh 5 2 4 0.50 tests/po.in
# sbatch config1.sh 12 0 10 0.16 tests/thesun.in

## Config2
# sbatch config2.sh 16 1 2 0.10 tests/tinkywinky.in
# sbatch config2.sh 5 3 5 0.50 tests/dipsy.in
# sbatch config2.sh 5 2 2 0.00 tests/lala.in
# sbatch config2.sh 5 2 4 0.50 tests/po.in
# sbatch config2.sh 12 0 10 0.16 tests/thesun.in

## Config3
# sbatch config3.sh 16 1 2 0.10 tests/tinkywinky.in
# sbatch config3.sh 5 3 5 0.50 tests/dipsy.in
# sbatch config3.sh 5 2 2 0.00 tests/lala.in
# sbatch config3.sh 5 2 4 0.50 tests/po.in
# sbatch config3.sh 12 0 10 0.16 tests/thesun.in


## We already ran this for you to get sequential outputs and times, no need to re-run
# ./generate_test_output.sh 16 1 2 0.10 tests/tinkywinky.in
# ./generate_test_output.sh 5 3 5 0.50 tests/dipsy.in
# ./generate_test_output.sh 5 2 2 0.00 tests/lala.in
# ./generate_test_output.sh 5 2 4 0.50 tests/po.in
# ./generate_test_output.sh 12 0 10 0.16 tests/thesun.in


