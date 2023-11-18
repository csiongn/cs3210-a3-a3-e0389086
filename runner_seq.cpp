#include "runner.hpp"
#include <fstream>
#include <queue>

  /*
   * ==================================================================================
   * | [START] DO NOT MODFIY THIS FILE, THIS IS THE REFERENCE SEQUENTIAL CODE [START] |
   * 
   * This is used to compile the reference sequential executable regardless of 
   * how you change runner.cpp.
   * ==================================================================================
   */
void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params) {
  if (rank == 0) {
    std::queue<task_t> task_queue;

    std::ifstream istrm(params.input_path, std::ios::binary);
    // Read initial tasks
    int count;
    istrm >> count;

    for (int i = 0; i < count; ++i) {
      task_t task;
      int type;
      istrm >> type >> task.arg_seed;
      task.type = static_cast<TaskType>(type);
      task.id = task.arg_seed;
      task.gen = 0;
      task_queue.push(task);
    }

    // Declare array to store generated descendant tasks
    int num_new_tasks = 0;
    std::vector<task_t> task_buffer(Nmax);
    while (!task_queue.empty()) {
      execute_task(stats, task_queue.front(), num_new_tasks, task_buffer);
      for (int i = 0; i < num_new_tasks; ++i) {
        task_queue.push(task_buffer[i]);
      }
      task_queue.pop();
    }
  }
}
