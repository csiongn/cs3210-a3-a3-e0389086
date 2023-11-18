#pragma once

#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory.h>
#include <mpi.h>
#include <vector>

/*
 * ========== MODIFY THESE AS REQUIRED ==========
 */
#ifndef DEBUG
  #define DEBUG 0
#endif
constexpr int CHILD_DEPTH = 1;

/*
 * ================ DO NOT MODIFY ================
 * ====== GLOBAL TASK GENERATION PARAMETERS ======
 */
extern int H, Nmin, Nmax;
extern float P;

/*
 * ================ DO NOT MODIFY ================
 * =============== TASK PARAMETERS ===============
 */
enum class TaskType { PRIME = 1, MATMULT, LCS, SHA, BITONIC };

constexpr int PRIME_MIN = (1 << 14) - 1;
constexpr int PRIME_MAX = (1 << 20) - 1;
constexpr int MATMULT_MIN = 200;
constexpr int MATMULT_MAX = 400;
constexpr int LCS_ALPH_MIN = 3;
constexpr int LCS_ALPH_MAX = 7;
constexpr int LCS_MIN = 8000;
constexpr int LCS_MAX = 32000;
constexpr int SHA_MIN = 65536;
constexpr int SHA_MAX = 1048576;
constexpr int BITONIC_MIN = 14;
constexpr int BITONIC_MAX = 21;

/*
 * ================ DO NOT MODIFY ================
 * =========== SHA256 MACROS AND TYPES ===========
 */
using BYTE = unsigned char;
using WORD = uint32_t;

constexpr WORD ROTLEFT(WORD a, WORD b) { return (a << b) | (a >> (32 - b)); }

constexpr WORD ROTRIGHT(WORD a, WORD b) { return (a >> b) | (a << (32 - b)); }

constexpr WORD CH(WORD x, WORD y, WORD z) { return (x & y) ^ (~x & z); }

constexpr WORD MAJ(WORD x, WORD y, WORD z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

constexpr WORD EP0(WORD x) {
  return ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22);
}

constexpr WORD EP1(WORD x) {
  return ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25);
}

constexpr WORD SIG0(WORD x) {
  return ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ (x >> 3);
}

constexpr WORD SIG1(WORD x) {
  return ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ (x >> 10);
}

constexpr std::array<WORD, 64> k{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

struct SHA256_CTX {
  std::array<BYTE, 64> data;
  WORD datalen;
  unsigned long long bitlen;
  std::array<WORD, 8> state;
};

// ============= END SHA-256 SECTION =============

// Common parameters for task generation, read through command line and distributed later through MPI
struct params_t {
  int H;
  int Nmin;
  int Nmax;
  float P;
  const char *input_path;
};

// MPI message definitions for params_t
// WARNING: for efficiency, we don't sent the input_path, so its value is undefined in a receiving process
constexpr int params_t_num_blocks = 4;
constexpr int params_t_lengths[params_t_num_blocks] = {1, 1, 1, 1};
constexpr MPI_Aint params_t_displs[params_t_num_blocks] = {
    offsetof(struct params_t, H),
    offsetof(struct params_t, Nmin),
    offsetof(struct params_t, Nmax),
    offsetof(struct params_t, P)};
inline const MPI_Datatype params_t_types[params_t_num_blocks] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT};

// TODO: replace with chrono
typedef struct timespec timespec_t;

// Our internal metrics datatype
struct metric_t {
  int rank;
  std::array<int, 5> completed;
  uint64_t total_time;
  uint64_t compute_time;
};

// MPI message definitions for our metrics
constexpr int metric_t_num_blocks = 4;
constexpr int metric_t_lengths[metric_t_num_blocks] = {1, 5, 1, 1}; 
constexpr MPI_Aint metric_t_displs[metric_t_num_blocks] = {
    offsetof(struct metric_t, rank),
    offsetof(struct metric_t, completed),
    offsetof(struct metric_t, total_time),
    offsetof(struct metric_t, compute_time)};
inline const MPI_Datatype metric_t_types[metric_t_num_blocks] = {MPI_INT, MPI_INT, MPI_UINT64_T, MPI_UINT64_T};

// All information required to execute a task with execute_task
struct task_t {
  uint32_t id;
  int gen;
  TaskType type;
  uint32_t arg_seed;
  uint32_t output;
  int num_dependencies;
  std::array<uint32_t, 4> dependencies;
  std::array<uint32_t, 4> masks;
};

template <typename T> class matrix {
  int n;
  T *_data;

public:
  matrix(int m, int n) : n(n) { _data = new T[m * n](); }
  ~matrix() { delete[] _data; }
  T &operator[](std::array<int, 2> idx) { return _data[idx[0] * n + idx[1]]; }
  const T &operator[](std::array<int, 2> idx) const {
    return _data[idx[0] * n + idx[1]];
  }
  T *data() { return _data; }
  const T *data() const { return _data; }
};

template <typename T> class vec {
  T *_data;

public:
  vec(int n) { _data = new T[n](); }
  ~vec() { delete[] _data; }
  T &operator[](int idx) { return _data[idx]; }
  const T &operator[](int idx) const { return _data[idx]; }
  T *data() { return _data; }
  const T *data() const { return _data; }
};

/*
 * ================ DO NOT MODIFY ================
 * === TASK GENERATION AND EXECUTION FUNCTIONS ===
 */
// Pseudo-random number generator
uint32_t get_next(uint32_t seed);

params_t parse_params(char *argv[]);
void set_generation_params(params_t &params);
uint64_t compute_interval(const timespec_t &start, const timespec_t &end);
void execute_task(metric_t &stats, task_t &task, int &num_generated,
                  std::vector<task_t> &children);
void generate_desc_tasks(task_t &root, uint32_t seed, int &num_generated,
                         std::vector<task_t> &children);

/*
 * ================ DO NOT MODIFY ================
 * ========= TASK PROCEDURES AND HELPERS =========
 * ======== SHOULD NOT BE INVOKED DIRECTLY =======
 */
uint32_t PRIME(uint32_t num);
uint32_t MATMULT(int m, int n, int p, uint32_t seed);
uint32_t LCS(int alph, int len1, int len2, uint32_t seed);
uint32_t BITONIC(int n, uint32_t seed);
std::array<char, 64> SHA(int len, uint32_t seed);

// Sub-routines for bitonic sort
void bitonic_compare_swap(vec<uint32_t> &a, int i, int j, int dir);
void bitonic_merge(vec<uint32_t> &a, int low, int cnt, int dir);
void bitonic_sort(vec<uint32_t> &a, int low, int cnt, int dir);

// Helper routines for SHA-256
void sha256_init(SHA256_CTX &ctx);
void sha256_update(SHA256_CTX &ctx, const vec<BYTE> &data, size_t len);
void sha256_final(SHA256_CTX &ctx, std::array<BYTE, 32> &hash);

/*
 * ================ DO NOT MODIFY ================
 * Useful helper routines for I/O
 */
void print_metrics(const metric_t &stats);
void print_combined_metrics(std::vector<metric_t> &metrics);
void print_task(const task_t &task, int num_desc);
void print_params(const params_t &params, int rank);