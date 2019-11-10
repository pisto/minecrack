#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <unordered_map>
extern "C" {
#define register
#include "cubiomes/generator.h"
#undef register
}

struct biome_position {
	int x, z;
	BiomeID biome;
	bool operator<(const biome_position& o) const {
		return std::make_tuple(x, z, biome) < std::make_tuple(o.x, o.z, o.biome);
	}
};

namespace cmdline {
extern bool verbose;
extern MCversion mcversion;
extern std::vector<biome_position> biome_positions;
}

constexpr const uint8_t LOW_SEED_BITS = 48;

std::vector<int64_t> check_biomes(uint64_t seed);
std::unordered_map<BiomeID, uint64_t> statistics(uint32_t seeds, uint32_t samples_per_seed);
