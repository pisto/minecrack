#pragma once

#include <cstdint>
#include <vector>
#include <csignal>

namespace cmdline {

extern bool verbose;
extern std::vector<int64_t> chunk_seed_offset;

}

constexpr const uint8_t LOW_SEED_BITS = 18;

inline int64_t slime_seed_offset(int chunkX, int chunkZ) {
	return int64_t(chunkX * chunkX * 4987142) + int64_t(chunkX * 5947611) + int64_t(chunkZ * chunkZ) * int64_t(4392871)
	       + int64_t(chunkZ * 389711);
}

inline int64_t mangle_seed(int64_t seed, int64_t offset) {
	return (seed + offset) ^ int64_t(987234911);
}

namespace CPU {

std::vector<uint32_t> lowbits_candidates();
std::vector<uint64_t> test_seeds(const std::vector<uint32_t>& lowbits_candidates);

}
