#pragma once

#include <cstdint>

/*
 * Smallest unsigned integer that can hold a value.
 */

#include <type_traits>

template<uint64_t max, typename = void> struct smallest_uint;
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max < 0x100)>> {
	using type = uint8_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x100 && max < 0x10000)>> {
	using type = uint16_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x10000 && max < 0x100000000ULL)>> {
	using type = uint32_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x100000000ULL)>> {
	using type = uint64_t;
};

/*
 * Skeleton C++ implementation of java.util.Random.
 */

struct JavaRandom {

	static constexpr const uint8_t generator_bits = 48;
	static constexpr const int64_t multiplier = 0x5DEECE66DLL, addend = 0xB;
	static constexpr const int JAVA_RAND_MAX = (1u << 31) - 1;

	template<typename T> JavaRandom(T seed): scrambled_seed(seed ^ multiplier) {}

	uint64_t current_seed() const { return scrambled_seed ^ multiplier; }

	int next(int bits) {
		scrambled_seed = int64_t(scrambled_seed) * multiplier + addend;
		return scrambled_seed >> (generator_bits - bits);
	}

	template<uint32_t bound, typename R = typename smallest_uint<bound - 1>::type> R nextInt() {
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
