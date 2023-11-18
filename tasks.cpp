#include "tasks.hpp"
#include <algorithm>

int H, Nmin, Nmax;
float P;

/*
 * ================ DO NOT MODIFY ================
 * Utility function to print out execution metrics on a single line
 */
void print_metrics(const metric_t &stats) {
  double compute_frac = (double)stats.compute_time / stats.total_time;
  printf("Rank %d: %" PRIu64 "ms of %" PRIu64
         "ms (%.5lf) - completed: %d %d %d %d %d\n",
         stats.rank, stats.compute_time, stats.total_time, compute_frac,
         stats.completed[0], stats.completed[1], stats.completed[2],
         stats.completed[3], stats.completed[4]);
}

/*
 * ================ DO NOT MODIFY ================
 * Utility function to print out summary of all execution metrics on a single line
 */
void print_combined_metrics(std::vector<metric_t> &metrics) {
    uint64_t total_compute_time = 0;
    uint64_t total_total_time = 0;
    std::array<int, 5> total_completed = {0};

    for (const metric_t &stats : metrics) {
        total_compute_time += stats.compute_time;
        total_total_time += stats.total_time;
        for (int i = 0; i < 5; i++) {
            total_completed[i] += stats.completed[i];
        }
    }

    double average_utilization = (double)total_compute_time / total_total_time;

    printf("Overall: %" PRIu64 "ms of %" PRIu64 "ms (%.5lf) - completed: %d %d %d %d %d\n",
               total_compute_time, total_total_time, average_utilization,
               total_completed[0], total_completed[1], total_completed[2],
               total_completed[3], total_completed[4]);
    printf("FINAL RUNTIME: %" PRIu64 "ms\n", metrics[0].total_time);
}



/*
 * ================ DO NOT MODIFY ================
 * Utility function to print out task information for execution traces
 */
void print_task(const task_t &task, int num_desc) {
  printf("EXECUTION TRACE: Task #%u: depth %d, type %d, seed %u, output %u (%d descendants)\n",
         task.id, task.gen,
         static_cast<std::underlying_type_t<TaskType>>(task.type),
         task.arg_seed, task.output, num_desc);
}

/*
 * ================ DO NOT MODIFY ================
 * Utility function to print out params info for verification
 */
void print_params(const params_t &params, int rank) {
  printf("Rank %d Params: H %d, Nmin %d, Nmax %d, P %f, Initial task filepath %s\n",
         rank, params.H, params.Nmin, params.Nmax, params.P, params.input_path);
}



/*
 * ================ DO NOT MODIFY ================
 * Simple, fast XORshift pseudorandom number generator
 * 32-bit state with period of 2^32 - 1; only fixed point is 0
 */
uint32_t get_next(uint32_t seed) {
  uint32_t x = seed;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x == 0 ? 1 : x;
}

/*
 * ================ DO NOT MODIFY ================
 * Parses argv into a parameter struct for broadcast with MPI; terminates if
 * any task generation parameter is invalid
 */
params_t parse_params(char *argv[]) {
  int error = 0;
  params_t params{
      atoi(argv[1]), atoi(argv[2]),
      atoi(argv[3]), static_cast<float>(atof(argv[4])),
      argv[5],
  };

  if (params.H < 0) {
    error = 1;
    fprintf(stderr, "Invalid H - should be non-negative, got %d!\n", params.H);
  }

  if (params.Nmin < 0 || params.Nmax < 0 || params.Nmin > params.Nmax) {
    error = 1;
    fprintf(stderr,
            "Invalid Nmin/Nmax - should be non-negative, got %d and %d!\n",
            params.Nmin, params.Nmax);
  }

  if (params.P < 0 || params.P > 1) {
    error = 1;
    fprintf(stderr, "Invalid P - should satisfy 0 <= P <= 1, got %.8f!\n",
            params.P);
  }

  if (error) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return params;
}

/*
 * ================ DO NOT MODIFY ================
 * Sets the global values for the task generation parameters
 */
void set_generation_params(params_t &params) {
  H = params.H;
  Nmin = params.Nmin;
  Nmax = params.Nmax;
  P = params.P;
}

/*
 * ================ DO NOT MODIFY ================
 * Computes the interval of time between two timestamps
 */
uint64_t compute_interval(const timespec_t &begin, const timespec_t &end) {
  uint64_t diff_sec = ((uint64_t)end.tv_sec) - begin.tv_sec;
  uint64_t tv_msec = 1000 * diff_sec + (end.tv_nsec - begin.tv_nsec) / 1000000;
  return tv_msec;
}

/*
 * ================ DO NOT MODIFY ================
 * Executes a given task, updating the execution metrics of the executing
 * process, then invokes generate_child_tasks to generate a random number of
 * new descendant tasks subject to provided task generation parameters
 */
void execute_task(metric_t &stats, task_t &task, int &num_generated,
                  std::vector<task_t> &children) {
  int rank;

  if (DEBUG == 2) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Rank %d started task #%u (type %d) with seed: %u\n", rank, task.id,
           static_cast<std::underlying_type_t<TaskType>>(task.type),
           task.arg_seed);
  }

  timespec_t start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  uint32_t seed = task.arg_seed;

  switch (task.type) {
  case TaskType::PRIME: {
    uint32_t num =
        (seed = get_next(seed)) % (PRIME_MAX - PRIME_MIN) + PRIME_MIN;
    task.output = PRIME(num);
    break;
  }
  case TaskType::MATMULT: {
    int m = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                (MATMULT_MAX - MATMULT_MIN) +
            MATMULT_MIN;
    int n = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                (MATMULT_MAX - MATMULT_MIN) +
            MATMULT_MIN;
    int p = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                (MATMULT_MAX - MATMULT_MIN) +
            MATMULT_MIN;
    task.output = MATMULT(m, n, p, seed);
    break;
  }
  case TaskType::LCS: {
    int alph = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                   (LCS_ALPH_MAX - LCS_ALPH_MIN) +
               LCS_ALPH_MIN;
    int len1 = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                   (LCS_MAX - LCS_MIN) +
               LCS_MIN;
    int len2 = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                   (LCS_MAX - LCS_MIN) +
               LCS_MIN;
    task.output = LCS(alph, len1, len2, seed);
    break;
  }
  case TaskType::SHA: {
    int len = (seed = get_next(seed)) % std::numeric_limits<int32_t>::max() %
                  (SHA_MAX - SHA_MIN) +
              SHA_MIN;
    auto result = SHA(len, seed);

    for (int i = 0; i < 4; i++) {
      task.output <<= 8;
      task.output ^= result[i];
    }
    break;
  }
  case TaskType::BITONIC: {
    int num_bits = (seed = get_next(seed)) %
                       std::numeric_limits<int32_t>::max() %
                       (BITONIC_MAX - BITONIC_MIN) +
                   BITONIC_MIN;
    task.output = BITONIC(num_bits, seed);
    break;
  }
  default:
    fprintf(stderr, "Invalid task type integer!\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Generate descendant tasks with output as the new seed
  generate_desc_tasks(task, task.output, num_generated, children);

  timespec_t end;
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t exec_time = compute_interval(start, end);

  // Update execution metrics
  stats.compute_time += exec_time;
  stats.completed[static_cast<std::underlying_type_t<TaskType>>(task.type) -
                  1]++;

  if (DEBUG == 1) {
    // Print full details of task to trace execution
    print_task(task, num_generated);
  } else if (DEBUG == 2) {
    printf("Rank %d completed task #%u (type %d) to %u in: %" PRIu64 "ms\n",
           rank, task.id,
           static_cast<std::underlying_type_t<TaskType>>(task.type),
           task.output, exec_time);
  }
}

/*
 * ================ DO NOT MODIFY ================
 * Iteratively generates a directed-acyclic graph of new descendant tasks
 * rooted at the current task
 */
void generate_desc_tasks(task_t &root, uint32_t seed, int &num_generated,
                         std::vector<task_t> &children) {
  // Zero out array of descendant tasks
  num_generated = 0;
  memset(children.data(), 0, children.size() * sizeof(task_t));

  if (root.gen >= H) {
    return;
  }

  num_generated = Nmin;
  for (int i = 0; i < Nmax - Nmin; i++) {
    // Generate a random number Z from (0, 1]
    float z = (seed = get_next(seed)) /
              static_cast<float>(std::numeric_limits<uint32_t>::max());
    if (z < P) {
      num_generated++;
    }
  }

  // Generate index pointers for partitions
  int num_partitions = H - root.gen > CHILD_DEPTH ? CHILD_DEPTH : H - root.gen;
  vec<int> partitions(num_partitions + 1);
  partitions[0] = 0;
  for (int i = 1; i < num_partitions; i++) {
    int remaining = num_generated - partitions[i - 1];
    int split_index = num_generated;
    if (remaining > 0) {
      split_index =
          (seed = get_next(seed)) % (remaining + 1) + partitions[i - 1];
    }

    if (split_index == partitions[i - 1] || split_index == num_generated) {
      i--;
      num_partitions--;
    } else {
      partitions[i] = split_index;
    }
  }
  partitions[num_partitions] = num_generated;

  if (DEBUG == 2) {
    printf("  Total: %d descendants, %d partitions\n", num_generated,
           num_partitions);
  }

  vec<int> track_all(
      *std::max_element(partitions.data(), partitions.data() + num_partitions));
  for (int p = 0; p < num_partitions; p++) {
    if (DEBUG == 2) {
      printf("    Partition %d: indices [%d, %d)\n", p, partitions[p],
             partitions[p + 1]);
    }

    // Generate base task params
    for (int i = partitions[p]; i < partitions[p + 1]; i++) {
      children[i].id = (seed = get_next(seed));
      children[i].gen = root.gen + 1;
      children[i].type = static_cast<TaskType>((seed = get_next(seed)) % 5 + 1);
      if (p == 0) {
        children[i].arg_seed = children[i].id;
      }
    }

    if (p == 0) {
      continue;
    }

    // Obtain dependencies for deep partitions
    int width = partitions[p] - partitions[p - 1];

    int max_total_rolls = width > 4 ? 4 : width;
    max_total_rolls = max_total_rolls > Nmax ? Nmax : max_total_rolls;
    // Generate dependencies of each task
    for (int i = partitions[p]; i < partitions[p + 1]; i++) {
      memset(track_all.data(), 0, partitions[p] * sizeof(int));

      // Roll for total number of dependencies
      children[i].num_dependencies = 1;
      if (max_total_rolls > 1) {
        children[i].num_dependencies +=
            (seed = get_next(seed)) % max_total_rolls;
      }

      if (DEBUG == 2) {
        printf("      Descendant %d: %d dependencies\n", i,
               children[i].num_dependencies);
      }

      // Decide primary dependency for current task
      int primary_dep = (seed = get_next(seed)) % width + partitions[p - 1];
      track_all[primary_dep] = 1;
      children[i].dependencies[0] = children[primary_dep].id;
      children[i].masks[0] = (seed = get_next(seed));

      if (DEBUG == 2) {
        printf("        Primary dep: desc %d (task #%u)\n", primary_dep,
               children[i].dependencies[0]);
      }

      // Generate secondary (extra) dependencies for current task
      for (int j = 1; j < children[i].num_dependencies; j++) {
        int secondary_dep = (seed = get_next(seed)) % partitions[p];
        while (track_all[secondary_dep] == 1) {
          secondary_dep = (seed = get_next(seed)) % partitions[p];
        }
        track_all[secondary_dep] = 1;
        children[i].dependencies[j] = children[secondary_dep].id;
        children[i].masks[j] = (seed = get_next(seed));

        if (DEBUG == 2) {
          printf("        Extra dep: desc %d (task #%u)\n", secondary_dep,
                 children[i].dependencies[j]);
        }
      }
    }
  }
}

/*
 * ================ DO NOT MODIFY ================
 * Task procedure for: PRIME <num>
 *
 * Counts the number of primes up to and including num by checking
 * primality of each candidate with the square-root method
 */
uint32_t PRIME(uint32_t num) {
  int flag, count = 0;

  for (uint32_t n = 2; n <= num; n++) {
    flag = 1;
    for (int i = 2; i <= sqrt(n); i++) {
      if (n % i == 0) {
        flag = 0;
        break;
      }
    }
    count += flag;
  }

  return count;
}

/*
 * ================ DO NOT MODIFY ================
 * Task procedure for: MATMULT <m> <n> <p>
 *
 * Multiplies the (m x n) matrix A with a (n x p) matrix B to obtain a (m x p)
 * result matrix C
 */
uint32_t MATMULT(int m, int n, int p, uint32_t seed) {
  matrix<int> A(m, n), B(n, p), C(m, p);

  // Generate random elements for matrices A and B
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[{i, j}] = seed = get_next(seed);
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      B[{i, j}] = seed = get_next(seed);
    }
  }

  // Actual matrix multiplication begins here
  for (int i = 0; i < m; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < p; j++) {
        C[{i, j}] += A[{i, k}] * B[{k, j}];
      }
    }
  }

  int row = (seed = get_next(seed)) % m;
  int col = (seed = get_next(seed)) % p;
  return C[{row, col}];
}

/*
 * ================ DO NOT MODIFY ================
 * Task procedure for: LCS <alph> <len1> <len2>
 *
 * Computes the Longest Common Subsequence (LCS) of two strings of length
 * len1 and len2 from an alphabet containing alph symbols
 */
uint32_t LCS(int alph, int len1, int len2, uint32_t seed) {
  vec<uint32_t> seq1(len1), seq2(len2);

  // Generate string of random symbols
  for (int i = 0; i < len1; i++) {
    seq1[i] = (seed = get_next(seed)) % alph;
  }

  for (int i = 0; i < len2; i++) {
    seq2[i] = (seed = get_next(seed)) % alph;
  }

  // Actual computation of LCS begins here
  matrix<int> mat(2, len2 + 1);
  mat[{1, 0}] = 0;

  int prev = 0;
  int next = 1;
  for (int i = 1; i <= len1; i++) {
    for (int j = 1; j <= len2; j++) {
      if (seq1[i - 1] == seq2[j - 1]) {
        mat[{next, j}] = mat[{prev, j - 1}] + 1;
      } else {
        uint32_t left = mat[{next, j - 1}];
        uint32_t down = mat[{prev, j}];
        mat[{next, j}] = left > down ? left : down;
      }
    }
    // Swap the previous and current row
    prev = prev ^ next;
    next = next ^ prev;
    prev = prev ^ next;
  }

  return mat[{prev, len2}];
}

/*
 * ================ DO NOT MODIFY ================
 * Task procedure for BITONIC <n>
 *
 * Sorts an array of 2^n integers in the range [1, 2^32 - 1] using bitonic sort
 */
uint32_t BITONIC(int n, uint32_t seed) {
  int length = 1 << n;
  vec<uint32_t> seq(length);

  // Generate random integers in input array
  for (int i = 0; i < length; i++) {
    seq[i] = seed = get_next(seed);
  }

  // Actual bitonic sort begins here
  bitonic_sort(seq, 0, 1 << n, 1);

  int idx = (seed = get_next(seed)) % length;
  return seq[idx];
}

/*
 * Helper routine for comparing and swapping a pair of integers in bitonic sort
 * according to a specified direction
 */
void bitonic_compare_swap(vec<uint32_t> &a, int i, int j, int dir) {
  if (dir == (a[i] > a[j])) {
    std::swap(a[i], a[j]);
  }
}

/*
 * Helper routine for merging bitonic sequences
 */
void bitonic_merge(vec<uint32_t> &a, int low, int cnt, int dir) {
  if (cnt > 1) {
    int k = cnt / 2;
    for (int i = low; i < low + k; i++) {
      bitonic_compare_swap(a, i, i + k, dir);
    }

    bitonic_merge(a, low, k, dir);
    bitonic_merge(a, low + k, k, dir);
  }
}

/*
 * Main bitonic sort routine
 */
void bitonic_sort(vec<uint32_t> &a, int low, int cnt, int dir) {
  if (cnt > 1) {
    int k = cnt / 2;
    bitonic_sort(a, low, k, 1);
    bitonic_sort(a, low + k, k, 0);
    bitonic_merge(a, low, cnt, dir);
  }
}

/*
 * ================ DO NOT MODIFY ================
 * Task procedure for SHA <str>
 *
 * Computes the SHA-256 message digest of the string str
 */
std::array<char, 64> SHA(int len, uint32_t seed) {
  vec<BYTE> data(len);

  // Generate random ASCII string with characters in range [48, 122]
  for (int i = 0; i < len; i++) {
    data[i] = 48 + ((seed = get_next(seed)) % (123 - 48));
  }

  SHA256_CTX payload;
  std::array<BYTE, 32> hash;
  std::array<char, 64> result;

  sha256_init(payload);
  sha256_update(payload, data, len);
  sha256_final(payload, hash);

  for (int i = 0; i < 32; i++) {
    sprintf(result.data() + 2 * i, "%02x", hash[i]);
  }

  return result;
}

void sha256_transform(SHA256_CTX &ctx, const std::array<BYTE, 64> &data) {
  WORD a, b, c, d, e, f, g, h, i, j, t1, t2;
  std::array<WORD, 64> m;

  for (i = 0, j = 0; i < 16; ++i, j += 4) {
    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) |
           (data[j + 3]);
  }

  for (; i < 64; ++i) {
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
  }

  a = ctx.state[0];
  b = ctx.state[1];
  c = ctx.state[2];
  d = ctx.state[3];
  e = ctx.state[4];
  f = ctx.state[5];
  g = ctx.state[6];
  h = ctx.state[7];

  for (i = 0; i < 64; ++i) {
    t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx.state[0] += a;
  ctx.state[1] += b;
  ctx.state[2] += c;
  ctx.state[3] += d;
  ctx.state[4] += e;
  ctx.state[5] += f;
  ctx.state[6] += g;
  ctx.state[7] += h;
}

void sha256_init(SHA256_CTX &ctx) {
  ctx.datalen = 0;
  ctx.bitlen = 0;
  ctx.state[0] = 0x6a09e667;
  ctx.state[1] = 0xbb67ae85;
  ctx.state[2] = 0x3c6ef372;
  ctx.state[3] = 0xa54ff53a;
  ctx.state[4] = 0x510e527f;
  ctx.state[5] = 0x9b05688c;
  ctx.state[6] = 0x1f83d9ab;
  ctx.state[7] = 0x5be0cd19;
}

void sha256_update(SHA256_CTX &ctx, const vec<BYTE> &data, size_t len) {
  WORD i;

  for (i = 0; i < len; ++i) {
    ctx.data[ctx.datalen] = data[i];
    ctx.datalen++;
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data);
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

void sha256_final(SHA256_CTX &ctx, std::array<BYTE, 32> &hash) {
  WORD i = ctx.datalen;

  // Pad whatever data is left in the buffer
  if (ctx.datalen < 56) {
    ctx.data[i++] = 0x80;
    while (i < 56) {
      ctx.data[i++] = 0x00;
    }
  } else {
    ctx.data[i++] = 0x80;
    while (i < 64) {
      ctx.data[i++] = 0x00;
    }
    sha256_transform(ctx, ctx.data);
    memset(ctx.data.data(), 0, 56);
  }

  // Append to the padding the total message's length in bits and transform
  ctx.bitlen += ctx.datalen * 8;
  ctx.data[63] = ctx.bitlen;
  ctx.data[62] = ctx.bitlen >> 8;
  ctx.data[61] = ctx.bitlen >> 16;
  ctx.data[60] = ctx.bitlen >> 24;
  ctx.data[59] = ctx.bitlen >> 32;
  ctx.data[58] = ctx.bitlen >> 40;
  ctx.data[57] = ctx.bitlen >> 48;
  ctx.data[56] = ctx.bitlen >> 56;
  sha256_transform(ctx, ctx.data);

  // Since this implementation uses little endian byte ordering and SHA uses
  // big endian, reverse all the bytes when copying the final state to the
  // output hash
  for (i = 0; i < 4; ++i) {
    hash[i] = (ctx.state[0] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 4] = (ctx.state[1] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 8] = (ctx.state[2] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 12] = (ctx.state[3] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 16] = (ctx.state[4] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 20] = (ctx.state[5] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 24] = (ctx.state[6] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 28] = (ctx.state[7] >> (24 - i * 8)) & 0x000000ff;
  }
}
