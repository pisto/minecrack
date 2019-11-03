#include <string>
#include <thread>
#include <set>
#include <regex>
#include <iostream>
#include <csignal>
#include <boost/program_options.hpp>
#include "utilities.hpp"
#include "utilities_cuda.cuh"
#include "minecrack-48bit.hpp"
#include "JavaRandom.cuh"

using namespace std;

volatile sig_atomic_t quit_requested = false;

/*
 * Get exceptions instead of asserts from boost.
 */
namespace boost {
void assertion_failed(const char* expr, const char* function, const char* file, long line) {
	throw logic_error("Boost assert failed: "s + expr + ", at " + file + ":" + to_string(line) + " in " + function);
}

void assertion_failed_msg(const char* expr, const char* msg, const char* function, const char* file, long line) {
	throw logic_error(
			"Boost assert failed ("s + msg + "): " + "" + expr + ", at " + file + ":" + to_string(line) + " in " +
			function);
}
}

namespace cmdline {

bool verbose;
uint64_t base_seed;
std::vector<int64_t> chunk_seed_offset;

};

int main(int argc, char** argv) try {
	for (int s: { SIGINT, SIGTERM }) signal(s, [](int) { quit_requested = true; });

	{
		using namespace boost::program_options;
		options_description options;
		options.add_options()
				("verbose,v", "print extra informations")
				("base,b", value(&cmdline::base_seed)->default_value(0), "start from this seed")
				("force,f", "do not bail out if too few chunks are provided");
		if (argc == 1) {
			cerr << "Usage: " << argv[0] << " [options] chunk1X:chunk1Z chunk2X:chunk2Z ..." << endl << "Options:"
			     << endl << options;
			return 0;
		}
		options_description options_with_chunks;
		options_with_chunks.add(options).add_options()("slime-chunk", value<vector<string>>());
		positional_options_description positional;
		positional.add("slime-chunk", -1);
		variables_map vm;
		try {
			store(command_line_parser(argc, argv).options(options_with_chunks).positional(positional).run(),
					vm);
			notify(vm);
		} catch (const boost::program_options::error& e) { throw invalid_argument(e.what()); }
		if (cmdline::base_seed >= 1ULL << JavaRandom::generator_bits)
			throw invalid_argument("argument to -b must be a 48 bits number");
		cmdline::verbose = vm.count("verbose");
		auto slimechunks_vector = vm["slime-chunk"].as<vector<string>>();
		set<string> slimechunks(slimechunks_vector.begin(), slimechunks_vector.end());
		regex slimechunk_regex("([\\-\\+]?\\d+)\\:([\\-\\+]?\\d+)", regex::ECMAScript | regex::optimize);
		smatch result;
		for (auto& chunkspec: slimechunks) {
			if (!regex_match(chunkspec, result, slimechunk_regex))
				throw invalid_argument("slime chunk coordinate " + chunkspec + " cannot be parsed");
			try {
				int chunkX = stoi(result[1]), chunkZ = stoi(result[2]);
				cmdline::chunk_seed_offset.push_back(chunkX * chunkX * 0x4c1906 + chunkX * 0x5ac0db
				                                + chunkZ * chunkZ * 0x4307a7L + (chunkZ * 0x5f24f ^ 0x3ad8025f));
			} catch (const out_of_range& o) {
				throw invalid_argument("slime chunk coordinate " + chunkspec + " cannot is out of int range");
			}
		}
		/*
		 * each slime chunk gives log2(10) bits of information on the 48bit seed of the world. At least 15 slime
		 * chunks must be found to have one or just few 48bit candidates.
		 */
		if (cmdline::chunk_seed_offset.size() < 13 && !vm.count("force"))
			throw invalid_argument("Too few slime chunks provided, at least 13 are needed (or use the -f option)");
		if (cmdline::chunk_seed_offset.size() < 15)
			cerr << "Warning: 15 or more slime chunks should be provided" << endl;
		if (cmdline::chunk_seed_offset.size() > GPU::MAX_SLIME_CHUNKS)
			throw invalid_argument("Too many slime chunks provided, max is " + to_string(GPU::MAX_SLIME_CHUNKS));
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0) && assertcu;
	if (cmdline::verbose) {
		char busid[50];
		cudaDeviceGetPCIBusId(busid, sizeof(busid) - 1, 0) && assertcu;
		cerr << "Using GPU " << props.name << ", clocks " << props.clockRate / 1000 << '/'
		     << props.memoryClockRate / 1000 << endl;
	}

	{
		uint8_t slimechunks_tot = cmdline::chunk_seed_offset.size();
		set_device_object(slimechunks_tot, GPU::chunk_seed_offset_len);
		cudaMemcpy(GPU::chunk_seed_offset, cmdline::chunk_seed_offset.data(), sizeof(int64_t) * slimechunks_tot,
				cudaMemcpyDefault) && assertcu;
	}
	int blocks, threads;
	//cudaOccupancyMaxPotentialBlockSize is not defined out of nvcc, no idea why
	cudaError cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, const void* func,
	                                             size_t dynamicSMemSize, int blockSizeLimit);
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, (void*)GPU::test_seeds, 0, 0) && assertcu;
	if (cmdline::verbose) cerr << "blocks:threads " << blocks << ':' << threads << endl;

	/*
	 * We subdivide work in sets of seeds of size GPU::BATCH_SIZE, and we test them in a single grid. The grid outputs
	 * the succeeding seeds to a circular buffer residing in host memory. A thread continuously tests for the presence
	 * of seeds and outputs them to cout.
	 * Several grids are launched concurrently, because a batch of seeds is divided evenly across threads on the GPU
	 * into smaller batches, but these batches might take a different total time to test. By launching multiple grids,
	 * uneven amounts of work across SMs should be masked.
	 */
	cudalist<uint64_t, true> passed_seeds(GPU::PASSED_BUFF_LEN, false);
	auto passed_empty_slot = 1ULL << JavaRandom::generator_bits;
	loopi(GPU::PASSED_BUFF_LEN) passed_seeds[i] = passed_empty_slot;
	struct {
		cudaStream_t s;
		completion c;
	} workgrids[4];
	for (auto& grid: workgrids) cudaStreamCreateWithFlags(&grid.s, cudaStreamNonBlocking) && assertcu;
	uint32_t passed_buffer_i = 0;
	volatile uint64_t* passed_seeds_volatile = *passed_seeds;

	cudaDeviceSynchronize() && assertcu;
	while (true) {
		bool grid_runs = false;
		for (auto& grid: workgrids) {
			while (passed_seeds_volatile[passed_buffer_i] != passed_empty_slot) {
				cout << passed_seeds_volatile[passed_buffer_i] << endl;
				passed_seeds_volatile[passed_buffer_i++] = passed_empty_slot;
				passed_buffer_i &= GPU::PASSED_BUFF_MASK;
			}
			if (!grid.c.ready()) {
				grid_runs = true;
				continue;
			}
			if (quit_requested) continue;
			GPU::launch_test_seeds(blocks, threads, grid.s, cmdline::base_seed, cmdline::base_seed + GPU::BATCH_SIZE, passed_seeds.devptr());
			grid.c.record(grid.s);
			cmdline::base_seed += GPU::BATCH_SIZE;
			if (cmdline::base_seed >= 1ULL << JavaRandom::generator_bits) quit_requested = true;
		}
		if (quit_requested && !grid_runs) break;
	}

	if (cmdline::base_seed < 1ULL << JavaRandom::generator_bits)
		cerr << "Restart with -b " << cmdline::base_seed << endl;

} catch (const invalid_argument& e) {
	cerr << "Invalid argument: " << e.what() << endl;
	return 1;
} catch (const exception& e) {
	cerr << e.what() << endl;
	return 2;
}
