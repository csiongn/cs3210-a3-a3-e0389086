#!/usr/bin/env python3

import os
import sys

def read_output(file):
    metrics = []
    for line in file:
        line = line.strip()
        if line.startswith("Overall"):
            return line
    print(f"XXX check.py TEST FAILURE: Your program did not return a line with 'Overall: ...' indicating the final results")
    sys.exit(1)

def extract_completed(metric):
    start = metric.find("completed:") + len("completed:")
    completed_values = metric[start:].strip().split()
    return list(map(int, completed_values))

def read_execution_trace(file):
    trace_lines = set()
    for line in file:
        line = line.strip()
        if line.startswith("EXECUTION TRACE:"):
            trace_lines.add(line)
    return trace_lines

def get_runtime(file):
    for line in file:
        if "FINAL RUNTIME:" in line:
            parts = line.split(":")
            if len(parts) == 2:
                runtime_str = parts[1].strip()
                if runtime_str.endswith("ms"):
                    runtime_ms = int(runtime_str[:-2])  # Remove "ms" and convert to integer
                    return runtime_ms
    print(f"XXX check.py TEST FAILURE: Your program did not return a line with 'FINAL RUNTIME: ...' indicating the final results")
    sys.exit(1)

def main():
    if len(sys.argv) < 5:
        print("Usage: ./check.py <H> <Nmin> <Nmax> <P> <input_file_path>")
        sys.exit(1)

    H = sys.argv[1]
    Nmin = sys.argv[2]
    Nmax = sys.argv[3]
    P = sys.argv[4]
    input_file = sys.argv[5]
    input_file_without_ext = os.path.splitext(input_file)[0]
    expected_output_file = f"{input_file_without_ext}_{H}_{Nmin}_{Nmax}_{P}.out"

    print(f">>> check.py running tests on your output against {expected_output_file} as a quick validation of your program.")
    print(">>> Note that check.py is not guaranteed to be complete or entirely correct! We will not intentionally include any errors, however.\n")

    try:
        with open(expected_output_file, 'r') as expected_output, sys.stdin as actual_output:
            # Read files
            expected_output_str = expected_output.readlines()
            actual_output_str = actual_output.readlines()
            # Get the Overall: string
            expected_metrics_str = read_output(expected_output_str)
            actual_metrics_str = read_output(actual_output_str)
            # Get num of tasks completed of each type
            expected_completed = extract_completed(expected_metrics_str)
            actual_completed = extract_completed(actual_metrics_str)

            # Execution metrics test
            if expected_completed == actual_completed:
                print("✓✓✓ check.py Execution Metrics test passed. The number of tasks of each type completed by your program matches what is expected.")
                print(f"\tExpected completed tasks:\t{expected_completed}")
                print(f"\tActual completed tasks:\t\t{actual_completed}")
                print()

                # Get the runtime string
                seq_runtime = get_runtime(expected_output_str)
                actual_runtime = get_runtime(actual_output_str)

                if os.environ.get('DEBUG', '0') != '0':
                    print("~~~ check.py WARNING: The speedup below is likely an underestimation of your speedup. Set DEBUG=0 or don't set it at all to get an accurate speedup calculation.")
                print(f"---> Sequential runtime:\t{seq_runtime} ms (extracted from reference output file {expected_output_file})")
                print(f"---> Your runtime:\t\t{actual_runtime} ms")
                print(f"---> Speedup:\t\t\t{seq_runtime/actual_runtime}x")
                print()

                # Check if we should continue to the execution trace test
                if os.environ.get('DEBUG', '0') != '1':
                    print("NOTE: You are not running in DEBUG=1 mode, so we cannot check the execution trace.")
                    print("Therefore, we cannot ensure complete correctness of your code.")
                    sys.exit(0)

                # Get the execution trace lines
                expected_trace = read_execution_trace(expected_output_str)
                actual_trace = read_execution_trace(actual_output_str)

                # Run the execution trace test
                if expected_trace == actual_trace:
                    print(f"✓✓✓ check.py Execution Trace test passed. Execution trace lines ({len(actual_trace)} lines) match the expected output ({len(expected_trace)} lines).")
                else:
                    print("✗✗✗ check.py TEST FAILURE: Execution Trace test failed. Lines do not match the expected output.")
                    print("\t Lines in expected output but not in actual output:")
                    for line in expected_trace - actual_trace:
                        print(line)
                    print("\t Lines in actual output but not in expected output:")
                    for line in actual_trace - expected_trace:
                        print(line)
                    sys.exit(1)

            else:
                print("✗✗✗ check.py TEST FAILURE: Execution Metrics test failed. The number of tasks of each type completed by your program does not match what is expected.")
                print(f"\tExpected completed tasks: {expected_completed}")
                print(f"\tActual completed tasks: {actual_completed}")
                sys.exit(1)


    except FileNotFoundError:
        print(f"✗✗✗ check.py FAILURE: Expected output file {expected_output_file} not found.")
        print(f"✗✗✗ ADVICE: You may want to run the command `./generate_test_output.sh {H} {Nmin} {Nmax} {P} {input_file}` to generate the expected output file.")
        sys.exit(1)
    except Exception as e:
        print(f"✗✗✗ check.py FAILURE: An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
