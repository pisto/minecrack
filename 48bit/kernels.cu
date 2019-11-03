#include "cub/iterator/cache_modified_input_iterator.cuh"
#include "utilities.hpp"
#include "utilities_cuda.cuh"
#include "JavaRandom.cuh"
#include "minecrack-48bit.hpp"

cudaError cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, const void *func, size_t dynamicSMemSize,
                                             int blockSizeLimit) {
	return cudaOccupancyMaxPotentialBlockSize<const void *>(minGridSize, blockSize, func, dynamicSMemSize,
			blockSizeLimit);
}

namespace GPU {

__device__ int64_t seed_offsets[MAX_SLIME_CHUNKS];
__constant__ uint8_t seed_offsets_len;

__global__ void test_seeds(uint64_t seeds_base, uint64_t seeds_max, uint64_t* passed_seeds) {
	static uint32_t passed_buffer_i = 0;
	cub::CacheModifiedInputIterator<cub::LOAD_LDG, int64_t> seed_offsets(GPU::seed_offsets);
	auto threads = gridDim.x * blockDim.x, threadidx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t seeds_total = seeds_max - seeds_base, seeds_portion = seeds_total / threads;
	seeds_base += seeds_portion * threadidx;
	seeds_max = seeds_base + seeds_portion + (threadidx == threads - 1 ? seeds_total % threads : 0);

	/*
	 * To flatten the flow control and minimize thread divergence, we run seed_offsets_len tests on some current_seed:
	 * when a test fails, the current_seed is immediately replaced with another one, otherwise if all tests succeed the
	 * seed is not immediately put into the output buffer but it is saved in good_seed. The value of good_seed is
	 * flushed to the output after targets_len iterations of the test. In this way the test loops has less control flow
	 * branches, and all of them can generally be implemented with fast predicates. The test loop is long exactly
	 * targets_len so that only one good_seed can be generated, otherwise multiple seeds could pass the tests and
	 * good_seed could be overwritten before flushing the result to output.
	 */
	constexpr const uint64_t bad_seed = 1ULL << JavaRandom::generator_bits;
	uint64_t current_seed = bad_seed, good_seed = bad_seed;
	uint32_t current_target = 0;
	while (current_seed < seeds_max) {
		for (uint8_t i = 0; i < seed_offsets_len; i++) {
			if (!current_target)
				//don't return here to avoid complicated flow control, out of bound results will be filtered later
				current_seed = seeds_base++;
			JavaRandom gen(current_seed + seed_offsets[i]);
			if (gen.nextInt<10>()) current_target = 0;
			else if (++current_target == seed_offsets_len) {
				current_target = 0;
				good_seed = current_seed;
			}
		}
		if (good_seed != bad_seed && good_seed < seeds_max) {
			passed_seeds[atomicAdd(&passed_buffer_i, 1) & PASSED_BUFF_MASK] = good_seed;
			good_seed = bad_seed;
		}
	}
}

void launch_test_seeds(int blocks, int thread, cudaStream_t s, uint64_t start, uint64_t end, uint64_t* passed_seeds) {
	test_seeds<<<blocks, thread, 0, s>>>(start, end, passed_seeds);
	cudaGetLastError() && assertcu;
}

}
