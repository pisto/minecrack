#include <memory>
#include <vector>
#include <algorithm>
#include "minecrack-16bit.hpp"

using namespace std;

namespace {
struct biome_checker {

	biome_checker(MCversion mcversion):
			stack(mcversion),
			mapdata(make_unique<int[]>(calcRequiredBuf(&stack.layers[L_VORONOI_ZOOM_1], 1, 1)))
			{}

	void setseed(int64_t seed) { applySeed(&stack, seed); }

	BiomeID getBiomeAt(int x, int z) {
		genArea(&stack.layers[L_VORONOI_ZOOM_1], mapdata.get(), x, z, 1, 1);
		return BiomeID(mapdata[0]);
	}

private:
	struct LayerStack_raii: LayerStack {
		LayerStack_raii(MCversion mcversion) { static_cast<LayerStack&>(*this) = setupGenerator(mcversion); }
		~LayerStack_raii() { freeGenerator(*this); }
	} stack;
	unique_ptr<int[]> mapdata;
};
}

#ifdef _OPENMP
#define openmp(x) _Pragma(#x)
#else
#define openmp(x)
#endif

vector<int64_t> check_biomes(uint64_t lowbits) {
	vector<int64_t> ret;
	openmp(omp parallel)
	{
		biome_checker checker(cmdline::mcversion);
		vector<int64_t> hits;
		openmp(omp for)
		for (uint32_t highbits = 0; highbits < 0x10000; highbits++) {
			int64_t seed = (int64_t(highbits) << LOW_SEED_BITS) | lowbits;
			checker.setseed(seed);
			if (all_of(cmdline::biome_positions.begin(), cmdline::biome_positions.end(),
					[seed,&checker](const biome_position& test) {
						return checker.getBiomeAt(test.x, test.z) == test.biome; }))
				hits.push_back(seed);
		}
		openmp(omp critical)
		ret.insert(ret.end(), hits.begin(), hits.end());
	}
	return ret;
}

unordered_map<BiomeID, uint64_t> statistics(uint32_t seeds, uint32_t samples_per_seed) {
	unordered_map<BiomeID, uint64_t> ret;
	openmp(omp parallel)
	{
		biome_checker checker(cmdline::mcversion);
		unordered_map<BiomeID, uint64_t> hits;
		openmp(omp for)
		for (uint32_t seed = 0; seed < seeds; seed++) {
			checker.setseed(seed);
			int position = -1000 * int(samples_per_seed / 2);
			for (auto sample = samples_per_seed; sample; sample--, position += 1000)
				hits[checker.getBiomeAt(position, position)]++;
		}
		openmp(omp critical)
		for (auto& s: hits) ret[s.first] += s.second;
	}
	return ret;
}
