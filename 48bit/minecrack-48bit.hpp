#pragma once

#include <cstdint>
#include <vector>
#include <csignal>

extern volatile sig_atomic_t quit_requested;

namespace cmdline {

extern bool verbose;
extern uint64_t base_seed;
extern std::vector<int64_t> seed_offsets;

}

#include "utilities_cuda.cuh"

namespace GPU {

constexpr const uint8_t MAX_SLIME_CHUNKS = 30;
constexpr const uint32_t PASSED_BUFF_LEN = 1024, PASSED_BUFF_MASK = PASSED_BUFF_LEN - 1;
constexpr const uint64_t BATCH_SIZE = 1ULL << 30;

extern __device__ int64_t seed_offsets[MAX_SLIME_CHUNKS];
extern __constant__ uint8_t seed_offsets_len;

__global__ void test_seeds(uint64_t start, uint64_t end, uint64_t* passed_buffer);
void launch_test_seeds(int blocks, int thread, cudaStream_t s, uint64_t start, uint64_t end, uint64_t* passed_seeds);

}