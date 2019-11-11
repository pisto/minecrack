#include <algorithm>
#include "JavaRandom.hpp"
#include "minecrack-48bit.hpp"

using namespace std;

#ifdef _OPENMP
#define openmp(x) _Pragma(#x)
#else
#define openmp(x)
#endif

vector<uint32_t> lowbits_candidates() {
	vector<uint32_t> ret;
	for (uint32_t lowbits = 0; lowbits < 1 << LOW_SEED_BITS; lowbits++)
		if (all_of(cmdline::chunk_seed_offset.begin(), cmdline::chunk_seed_offset.end(),
				[lowbits](int64_t offset) {
					JavaRandom gen(mangle_seed(lowbits, offset));
					//detect modulo bias condition as in nextInt<bound>()
					int r = gen.next(31);
					constexpr const int BAD_RANDOM = JavaRandom::JAVA_RAND_MAX - JavaRandom::JAVA_RAND_MAX % 10;
					return r >= BAD_RANDOM || r % 2 == 0;
				}))
			ret.push_back(lowbits);
	return ret;
}

vector<uint64_t> test_seeds(const vector<uint32_t>& lowbits_candidates) {
	vector<uint64_t> ret;
	openmp(omp parallel)
	{
		vector<uint64_t> hits;
		openmp(omp for)
		for (uint64_t highbits = 0; highbits < 1ull << JavaRandom::generator_bits; highbits += 1 << LOW_SEED_BITS)
			for (auto lowbits: lowbits_candidates) {
				auto seed = highbits | lowbits;
				if (all_of(cmdline::chunk_seed_offset.begin(), cmdline::chunk_seed_offset.end(),
						[seed](int64_t offset) {
							JavaRandom gen(mangle_seed(seed, offset));
							return !gen.nextInt<10>();
						}))
					hits.push_back(seed);
			}
		openmp(omp critical)
		ret.insert(ret.end(), hits.begin(), hits.end());
	}
	return ret;

}
