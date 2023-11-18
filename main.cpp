#include "runner.hpp"
#include "tasks.hpp"

int main(int argc, char *argv[]) {
  /*
   * =======================================================================
   * | [START]             DO NOT MODFIY THIS PORTION              [START] |
   * =======================================================================
   */
  int rank, num_procs;
  params_t params;

  // Initialise MPI processes and their ranks
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /*
   * Create MPI datatypes from general datatypes, displacement and block sizes
   * Constant arrays initialised in tasks.h - beware of C structure packing!
   */
  MPI_Datatype MPI_PARAMS_T, MPI_METRIC_T;

  MPI_Type_create_struct(params_t_num_blocks, params_t_lengths, params_t_displs, params_t_types,
                         &MPI_PARAMS_T);
  MPI_Type_create_struct(metric_t_num_blocks, metric_t_lengths, metric_t_displs, metric_t_types,
                         &MPI_METRIC_T);

  MPI_Type_commit(&MPI_PARAMS_T);
  MPI_Type_commit(&MPI_METRIC_T);

  /*
   * By default, mpirun directs stdin to the rank 0 process and directs
   * standard input to /dev/null on all other processes
   */
  if (rank == 0) {
    if (argc != 6) {
      printf("Syntax: ./distr-sched <H> <Nmin> <Nmax> <P> <input_file_path>\n");
      printf("ERROR: incorrect number of command line arguments\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Parse task generation parameters from command-line arguments
    params = parse_params(argv);
  }

  /*
   * Broadcast task generation parameters then set them as globals locally
   * All other processes block here until rank 0 process executes broadcast
   */
  MPI_Bcast(&params, 1, MPI_PARAMS_T, 0, MPI_COMM_WORLD);
  set_generation_params(params);
  
  // Start tracking execution metrics for each MPI process
  timespec_t start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  metric_t stats = {rank, {0, 0, 0, 0, 0}, 1, 0};

  /*
   * =======================================================================
   * | [END]              OK TO MODIFY CODE AFTER HERE               [END] |
   * =======================================================================
   */
  run_all_tasks(rank, num_procs, stats, params);

  /*
   * =======================================================================
   * | [START]             DO NOT MODFIY THIS PORTION              [START] |
   * =======================================================================
   */
  MPI_Barrier(MPI_COMM_WORLD);

  // Complete tracking execution metrics for each process
  timespec_t end;
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats.total_time += compute_interval(start, end);

  if (rank == 0) {
    std::vector<metric_t> metrics(num_procs);
    MPI_Gather(&stats, 1, MPI_METRIC_T, metrics.data(), 1, MPI_METRIC_T, 0,
                MPI_COMM_WORLD);

    // Print execution metrics by rank order
    printf("========================== START EXECUTION METRIC BLOCK FOR VERIFICATION ==========================\n");
    for (const auto &metric : metrics) {
      print_metrics(metric);
    }

    // Sum and print combined metrics
    print_combined_metrics(metrics);
    printf("========================== END EXECUTION METRIC BLOCK FOR VERIFICATION ==========================\n");

  } else {
    MPI_Gather(&stats, 1, MPI_METRIC_T, NULL, 0, MPI_DATATYPE_NULL, 0,
                MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Terminate MPI execution environment
  MPI_Finalize();

  return 0;
  /*
   * =======================================================================
   * | [END]                  END OF SKELETON CODE                   [END] |
   * =======================================================================
   */
}
