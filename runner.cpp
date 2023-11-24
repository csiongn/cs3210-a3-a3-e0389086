// runner.cpp
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <queue>
#include <vector>
#include "runner.hpp"

// CONSTANTS AND TAGS
int MASTER_PROCESS = 0;
int INITIAL_SIGNAL_TAG = 0;
int READY_SIGNAL_TAG = 1;
int TASK_SIGNAL_TAG = 2;

// MASTER FUNCTIONS

// Function to distribute tasks to different processors
void distribute_tasks(std::queue<task_t>& master_queue, int num_procs, int rank, params_t &params) {
    int count = 0;
    std::ifstream istrm(params.input_path, std::ios::binary);
    istrm >> count;

    for (int i = 0; i < count; ++i) {
        task_t task;
        int type;
        istrm >> type >> task.arg_seed;
        task.type = static_cast<TaskType>(type);
        task.id = task.arg_seed;
        task.gen = 0;

        // Determine the recipient processor for the task
        int target_rank = i % (num_procs - 1);
        // Send the task to the target processor
        MPI_Send(&task, sizeof(task_t), MPI_BYTE, target_rank + 1, INITIAL_SIGNAL_TAG, MPI_COMM_WORLD);
    }
    istrm.close();

    // Send an "exit task" to each rank
    task_t exit_command;
    exit_command.id = UINT32_MAX; // Using a special value to indicate an exit task
    for (int i = 1; i < num_procs; ++i) {
        MPI_Send(&exit_command, sizeof(task_t), MPI_BYTE, i, INITIAL_SIGNAL_TAG, MPI_COMM_WORLD);
    }
}

// Redistribute tasks to ready workers
void redistribute_tasks(std::queue<task_t>& master_queue, std::vector<bool>& is_worker_ready, int num_procs) {
    for (int rank = 1; rank < num_procs; ++rank) {
        if (is_worker_ready[rank - 1] && !master_queue.empty()) {
            task_t task = master_queue.front();
            master_queue.pop();
            MPI_Send(&task, sizeof(task_t), MPI_BYTE, rank, TASK_SIGNAL_TAG, MPI_COMM_WORLD);
            is_worker_ready[rank - 1] = false;
        }
    }
}

// Send an exit signal to all worker processes
void send_exit_signal_to_all_workers(int num_procs) {
    task_t exit_command;
    exit_command.id = UINT32_MAX; // Special value to indicate an exit command

    for (int rank = 1; rank < num_procs; ++rank) {
        MPI_Send(&exit_command, sizeof(task_t), MPI_BYTE, rank, TASK_SIGNAL_TAG, MPI_COMM_WORLD);
    }
}

bool receive_task_from_worker(std::queue<task_t> &master_queue, MPI_Request &req) {
    static task_t received_task;
    MPI_Irecv(&received_task, sizeof(task_t), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &req);
    
    int flag;
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        master_queue.push(received_task);
        return true;
    }
    return false;
}

bool receive_ready_signal(std::vector<bool>& is_worker_ready, MPI_Request &req) {
    static int worker_rank;
    MPI_Irecv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, READY_SIGNAL_TAG, MPI_COMM_WORLD, &req);

    int flag;
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        is_worker_ready[worker_rank - 1] = true;
        return true;
    }
    return false;
}

void manage_worker_signals_and_tasks(std::queue<task_t>& master_queue, int num_procs) {
    std::vector<bool> is_worker_ready(num_procs - 1, false);
    MPI_Request task_req, ready_req;

    // Initialize non-blocking receives for tasks and ready signals
    static task_t received_task;
    static int worker_rank;
    MPI_Irecv(&received_task, sizeof(task_t), MPI_BYTE, MPI_ANY_SOURCE, TASK_SIGNAL_TAG, MPI_COMM_WORLD, &task_req);
    MPI_Irecv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, READY_SIGNAL_TAG, MPI_COMM_WORLD, &ready_req);

    while (true) {
        // Check for task completion
        int task_flag;
        MPI_Test(&task_req, &task_flag, MPI_STATUS_IGNORE);
        if (task_flag) {
            master_queue.push(received_task);
            MPI_Irecv(&received_task, sizeof(task_t), MPI_BYTE, MPI_ANY_SOURCE, TASK_SIGNAL_TAG, MPI_COMM_WORLD, &task_req);
        }

        // Check for ready signal completion
        int ready_flag;
        MPI_Test(&ready_req, &ready_flag, MPI_STATUS_IGNORE);
        if (ready_flag) {
            is_worker_ready[worker_rank - 1] = true;
            MPI_Irecv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, READY_SIGNAL_TAG, MPI_COMM_WORLD, &ready_req);
        }

        redistribute_tasks(master_queue, is_worker_ready, num_procs);

        // Check if all tasks are done and all workers are ready
        if (master_queue.empty() && std::all_of(is_worker_ready.begin(), is_worker_ready.end(), [](bool ready) { return ready; })) {
            send_exit_signal_to_all_workers(num_procs);
            break;
        }
    }

    MPI_Request_free(&task_req);
    MPI_Request_free(&ready_req);
}

// WORKER FUNCTIONS

void receive_initial_tasks(std::queue<task_t> &local_queue, int rank)
{
        while (true) {
            task_t task;
            MPI_Status status;
            MPI_Recv(&task, sizeof(task_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

            if (task.id == UINT32_MAX) break; // Exit loop if the exit command is received

            local_queue.push(task);
        }

}

void send_task_to_master(const task_t &task) {
    MPI_Send(&task, sizeof(task_t), MPI_BYTE, MASTER_PROCESS, TASK_SIGNAL_TAG, MPI_COMM_WORLD);
}

// Send ready signal from worker
void send_ready_signal(int rank) {
    int signal = rank;
    MPI_Send(&signal, 1, MPI_INT, MASTER_PROCESS, READY_SIGNAL_TAG, MPI_COMM_WORLD);
}

void run_worker_tasks(int rank, std::queue<task_t> &local_queue, metric_t &stats) {
    bool isReadySent = false;

    while (true) {
        if (!local_queue.empty()) {
            task_t current_task = local_queue.front();
            local_queue.pop();

            if (current_task.id == UINT32_MAX) {
                // Received an exit command
                break;
            }

            int num_new_tasks = 0;
            std::vector<task_t> task_buffer(Nmax);
            execute_task(stats, current_task, num_new_tasks, task_buffer);
            for (int i = 0; i < num_new_tasks; ++i) {
                if (!local_queue.empty()) {
                    send_task_to_master(task_buffer[i]);
                } else {
                    local_queue.push(task_buffer[i]);
                }
            }
        } else {
            if (!isReadySent) {
                send_ready_signal(rank);
                isReadySent = true;
            }

            // Poll for new tasks
            MPI_Status status;
            task_t new_task;
            MPI_Recv(&new_task, sizeof(task_t), MPI_BYTE, 0, TASK_SIGNAL_TAG, MPI_COMM_WORLD, &status);

            if (new_task.id == UINT32_MAX) {
                // Exit command received
                break;
            } else {
                // New task received, reset ready status and process task
                isReadySent = false;
                local_queue.push(new_task);
            }
        }
    }
}

// MAIN CONTROL

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params) {
    std::queue<task_t> local_queue;

    if (rank == 0) {
        distribute_tasks(local_queue, num_procs, rank, params);
        manage_worker_signals_and_tasks(local_queue, num_procs);
    } else {
        receive_initial_tasks(local_queue, rank);
        run_worker_tasks(rank, local_queue, stats);
    }
}
