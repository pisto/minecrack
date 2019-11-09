#include <string>
#include <set>
#include <map>
#include <regex>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <boost/program_options.hpp>
#include "minecrack-16bit.hpp"

using namespace std;

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
MCversion mcversion;
set<biome_position> biome_positions;
}

extern "C" {
Biome biomes[256];
}

namespace {

map<BiomeID, string> biomeID2name = {
#define biomeID2name(x) { x, #x }
		biomeID2name(ocean),
		biomeID2name(plains),
		biomeID2name(desert),
		biomeID2name(mountains),
		biomeID2name(forest),
		biomeID2name(taiga),
		biomeID2name(swamp),
		biomeID2name(river),
		biomeID2name(nether),
		biomeID2name(the_end),
		biomeID2name(frozen_ocean),
		biomeID2name(frozen_river),
		biomeID2name(snowy_tundra),
		biomeID2name(snowy_mountains),
		biomeID2name(mushroom_fields),
		biomeID2name(mushroom_field_shore),
		biomeID2name(beach),
		biomeID2name(desert_hills),
		biomeID2name(wooded_hills),
		biomeID2name(taiga_hills),
		biomeID2name(mountain_edge),
		biomeID2name(jungle),
		biomeID2name(jungle_hills),
		biomeID2name(jungle_edge),
		biomeID2name(deep_ocean),
		biomeID2name(stone_shore),
		biomeID2name(snowy_beach),
		biomeID2name(birch_forest),
		biomeID2name(birch_forest_hills),
		biomeID2name(dark_forest),
		biomeID2name(snowy_taiga),
		biomeID2name(snowy_taiga_hills),
		biomeID2name(giant_tree_taiga),
		biomeID2name(giant_tree_taiga_hills),
		biomeID2name(wooded_mountains),
		biomeID2name(savanna),
		biomeID2name(savanna_plateau),
		biomeID2name(badlands),
		biomeID2name(wooded_badlands_plateau),
		biomeID2name(badlands_plateau),
		biomeID2name(small_end_islands),
		biomeID2name(end_midlands),
		biomeID2name(end_highlands),
		biomeID2name(end_barrens),
		biomeID2name(warm_ocean),
		biomeID2name(lukewarm_ocean),
		biomeID2name(cold_ocean),
		biomeID2name(deep_warm_ocean),
		biomeID2name(deep_lukewarm_ocean),
		biomeID2name(deep_cold_ocean),
		biomeID2name(deep_frozen_ocean),
		biomeID2name(sunflower_plains),
		biomeID2name(desert_lakes),
		biomeID2name(gravelly_mountains),
		biomeID2name(flower_forest),
		biomeID2name(taiga_mountains),
		biomeID2name(swamp_hills),
		biomeID2name(ice_spikes),
		biomeID2name(modified_jungle),
		biomeID2name(modified_jungle_edge),
		biomeID2name(tall_birch_forest),
		biomeID2name(tall_birch_hills),
		biomeID2name(dark_forest_hills),
		biomeID2name(snowy_taiga_mountains),
		biomeID2name(giant_spruce_taiga),
		biomeID2name(giant_spruce_taiga_hills),
		biomeID2name(modified_gravelly_mountains),
		biomeID2name(shattered_savanna),
		biomeID2name(shattered_savanna_plateau),
		biomeID2name(eroded_badlands),
		biomeID2name(modified_wooded_badlands_plateau),
		biomeID2name(modified_badlands_plateau),
		biomeID2name(bamboo_jungle),
		biomeID2name(bamboo_jungle_hills)
#undef  biomeID2name
};

const map<string, BiomeID> name2biomeID = {
#define name2biomeID(x) { #x, x }
		name2biomeID(ocean),
		name2biomeID(plains),
		name2biomeID(desert),
		name2biomeID(mountains),
		name2biomeID(extremeHills),
		name2biomeID(forest),
		name2biomeID(taiga),
		name2biomeID(swamp),
		name2biomeID(swampland),
		name2biomeID(river),
		name2biomeID(nether),
		name2biomeID(hell),
		name2biomeID(the_end),
		name2biomeID(sky),
		name2biomeID(frozen_ocean),
		name2biomeID(frozenOcean),
		name2biomeID(frozen_river),
		name2biomeID(frozenRiver),
		name2biomeID(snowy_tundra),
		name2biomeID(icePlains),
		name2biomeID(snowy_mountains),
		name2biomeID(iceMountains),
		name2biomeID(mushroom_fields),
		name2biomeID(mushroomIsland),
		name2biomeID(mushroom_field_shore),
		name2biomeID(mushroomIslandShore),
		name2biomeID(beach),
		name2biomeID(desert_hills),
		name2biomeID(desertHills),
		name2biomeID(wooded_hills),
		name2biomeID(forestHills),
		name2biomeID(taiga_hills),
		name2biomeID(taigaHills),
		name2biomeID(mountain_edge),
		name2biomeID(extremeHillsEdge),
		name2biomeID(jungle),
		name2biomeID(jungle_hills),
		name2biomeID(jungleHills),
		name2biomeID(jungle_edge),
		name2biomeID(jungleEdge),
		name2biomeID(deep_ocean),
		name2biomeID(deepOcean),
		name2biomeID(stone_shore),
		name2biomeID(stoneBeach),
		name2biomeID(snowy_beach),
		name2biomeID(coldBeach),
		name2biomeID(birch_forest),
		name2biomeID(birchForest),
		name2biomeID(birch_forest_hills),
		name2biomeID(birchForestHills),
		name2biomeID(dark_forest),
		name2biomeID(roofedForest),
		name2biomeID(snowy_taiga),
		name2biomeID(coldTaiga),
		name2biomeID(snowy_taiga_hills),
		name2biomeID(coldTaigaHills),
		name2biomeID(giant_tree_taiga),
		name2biomeID(megaTaiga),
		name2biomeID(giant_tree_taiga_hills),
		name2biomeID(megaTaigaHills),
		name2biomeID(wooded_mountains),
		name2biomeID(extremeHillsPlus),
		name2biomeID(savanna),
		name2biomeID(savanna_plateau),
		name2biomeID(savannaPlateau),
		name2biomeID(badlands),
		name2biomeID(mesa),
		name2biomeID(wooded_badlands_plateau),
		name2biomeID(mesaPlateau_F),
		name2biomeID(badlands_plateau),
		name2biomeID(mesaPlateau),
		name2biomeID(small_end_islands),
		name2biomeID(end_midlands),
		name2biomeID(end_highlands),
		name2biomeID(end_barrens),
		name2biomeID(warm_ocean),
		name2biomeID(warmOcean),
		name2biomeID(lukewarm_ocean),
		name2biomeID(lukewarmOcean),
		name2biomeID(cold_ocean),
		name2biomeID(coldOcean),
		name2biomeID(deep_warm_ocean),
		name2biomeID(warmDeepOcean),
		name2biomeID(deep_lukewarm_ocean),
		name2biomeID(lukewarmDeepOcean),
		name2biomeID(deep_cold_ocean),
		name2biomeID(coldDeepOcean),
		name2biomeID(deep_frozen_ocean),
		name2biomeID(frozenDeepOcean),
		name2biomeID(the_void),
		name2biomeID(sunflower_plains),
		name2biomeID(desert_lakes),
		name2biomeID(gravelly_mountains),
		name2biomeID(flower_forest),
		name2biomeID(taiga_mountains),
		name2biomeID(swamp_hills),
		name2biomeID(ice_spikes),
		name2biomeID(modified_jungle),
		name2biomeID(modified_jungle_edge),
		name2biomeID(tall_birch_forest),
		name2biomeID(tall_birch_hills),
		name2biomeID(dark_forest_hills),
		name2biomeID(snowy_taiga_mountains),
		name2biomeID(giant_spruce_taiga),
		name2biomeID(giant_spruce_taiga_hills),
		name2biomeID(modified_gravelly_mountains),
		name2biomeID(shattered_savanna),
		name2biomeID(shattered_savanna_plateau),
		name2biomeID(eroded_badlands),
		name2biomeID(modified_wooded_badlands_plateau),
		name2biomeID(modified_badlands_plateau),
		name2biomeID(bamboo_jungle),
		name2biomeID(bamboo_jungle_hills)
#undef  name2biomeID
};

const map<string, MCversion> mcversions = {
		{ "1.7", MC_1_7 },
		{ "1.8", MC_1_8 },
		{ "1.9", MC_1_9 },
		{ "1.10", MC_1_10 },
		{ "1.11", MC_1_11 },
		{ "1.12", MC_1_12 },
		{ "1.13", MC_1_13 },
		{ "1.14", MC_1_14 },
		{ "BE", MCBE }
};
}

int main(int argc, char** argv) try {
	bool do_statistics;
	{
		using namespace boost::program_options;
		options_description options;
		string mcversion;
		options.add_options()
				("mcversion,m", value(&mcversion)->required(), "Minecraft version (1.# or BE)")
				("statistics", "print biome statistics")
				("verbose,v", "print extra informations");
		if (argc == 1) {
			cerr << "Usage: " << argv[0] << " [options] coord1X:coord1Z:biome coord2X:coord2Z:biome ..." << endl << "Options:"
			     << endl << options << endl << "Biome names:" << endl;
			for (auto& biome: name2biomeID) cerr << biome.first << endl;
			cerr << endl << "Minecraft versions:" << endl;
			for (auto& v: mcversions) cerr << v.first << endl;
			return 0;
		}
		options_description options_with_biomes;
		options_with_biomes.add(options).add_options()("biome", value<vector<string>>());
		positional_options_description positional;
		positional.add("biome", -1);
		variables_map vm;
		try {
			store(command_line_parser(argc, argv).options(options_with_biomes).positional(positional).run(),
					vm);
			notify(vm);
		} catch (const boost::program_options::error& e) { throw invalid_argument(e.what()); }
		cmdline::verbose = vm.count("verbose");
		do_statistics = vm.count("statistics");
		try { cmdline::mcversion = mcversions.at(mcversion); }
		catch (const out_of_range&) {
			throw invalid_argument("Minecraft version " + mcversion + " is not recognized");
		}
		if (!vm.count("biome")) {
			if (!do_statistics) throw invalid_argument("You must specify some biome coordinates");
		} else {
			auto biomes = vm["biome"].as<vector<string>>();
			regex biome_regex("([\\-\\+]?\\d+)\\:([\\-\\+]?\\d+):([a-z_]+)", regex::ECMAScript | regex::optimize);
			smatch result;
			for (auto& chunkspec: biomes) {
				if (!regex_match(chunkspec, result, biome_regex))
					throw invalid_argument("chunk coordinate and biome " + chunkspec + " cannot be parsed");
				int coordX, coordZ;
				BiomeID biome;
				try { coordX = stoi(result[1]), coordZ = stoi(result[2]); }
				catch (const out_of_range& o) {
					throw invalid_argument("biome coordinate " + chunkspec + " cannot is out of int range");
				}
				try { biome = name2biomeID.at(result[3]); }
				catch (const out_of_range& o) {
					throw invalid_argument("biome name " + string(result[3]) + " is not valid");
				}
				cmdline::biome_positions.insert({ coordX, coordZ, biome });
			}
		}
	}

	initBiomes();
	if (do_statistics) {
		vector<tuple<uint64_t, BiomeID, string>> ranking;
		double total = 0;
		for (auto& s: ::statistics(0x10000, 20)) {
			ranking.emplace_back(s.second, s.first, biomeID2name[s.first]);
			total += s.second;
		}
		sort(ranking.begin(), ranking.end());
		cout << fixed << setprecision(10);
		for (auto it = ranking.rbegin(); it != ranking.rend(); it++)
			cout << get<uint64_t>(*it) / total << ',' << int(get<BiomeID>(*it)) << ',' << get<string>(*it)
			     << endl;
		return 0;
	}

	uint64_t lower;
	cin.exceptions(ios::badbit | ios::failbit);
	while (!cin.eof() && cin >> lower) {
		if (lower > 1ull << 48) throw invalid_argument("Invalid input " + to_string(lower));
		auto seeds = check_biomes(lower);
		sort(seeds.begin(), seeds.end());
		for (auto s: seeds) cout << s;
	}

} catch (const invalid_argument& e) {
	cerr << "Invalid argument: " << e.what() << endl;
	return 1;
} catch (const exception& e) {
	cerr << e.what() << endl;
	return 2;
}
