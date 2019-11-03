#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "utilities.hpp"

/*
 * Skeleton C++ implementation of java.util.Random.
 */

struct JavaRandom {

	static constexpr const uint8_t generator_bits = 48;
	static constexpr const int64_t multiplier = 0x5DEECE66DLL, addend = 0xB;
	static constexpr const int JAVA_RAND_MAX = (1u << 31) - 1;

	template<typename T> __host__ __device__ JavaRandom(T seed): scrambled_seed(seed ^ multiplier) {}

	__host__ __device__ uint64_t current_seed() const { return scrambled_seed ^ multiplier; }

	__host__ __device__ [[gnu::always_inline]] int next(int bits) {
		scrambled_seed = int64_t(scrambled_seed) * multiplier + addend;
		return scrambled_seed >> (generator_bits - bits);
	}

	template<uint32_t bound, typename R = typename smallest_uint<bound - 1>::type> __host__ __device__ R nextInt() {
		int r = next(31);
		if (!(bound & (bound - 1))) return r & (bound - 1);
		constexpr const int BAD_RANDOM = JAVA_RAND_MAX - JAVA_RAND_MAX % bound;
		while (r >= BAD_RANDOM) r = next(31);
		using small_bound_type = typename smallest_uint<bound>::type;
		return r % small_bound_type(bound);
	}

private:
	uint64_t scrambled_seed : generator_bits;

};
