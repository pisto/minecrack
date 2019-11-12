package com.pisto;

import amidst.logging.AmidstLogger;
import amidst.mojangapi.file.DotMinecraftDirectoryNotFoundException;
import amidst.mojangapi.file.MinecraftInstallation;
import amidst.mojangapi.minecraftinterface.MinecraftInterface;
import amidst.mojangapi.minecraftinterface.MinecraftInterfaceCreationException;
import amidst.mojangapi.minecraftinterface.MinecraftInterfaceException;
import amidst.mojangapi.minecraftinterface.MinecraftInterfaces;
import amidst.mojangapi.world.WorldType;
import amidst.parsing.FormatException;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.ParseException;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BiomeSeedTest {

    static HashMap<String, Integer> biome_names = new HashMap<String, Integer>();
    static HashMap<Integer, Double> probability = new HashMap<Integer, Double>();
    static {
        biome_names.put("badlands", 37);
        biome_names.put("badlands_plateau", 39);
        biome_names.put("bamboo_jungle", 168);
        biome_names.put("bamboo_jungle_hills", 169);
        biome_names.put("beach", 16);
        biome_names.put("birch_forest", 27);
        biome_names.put("birch_forest_hills", 28);
        biome_names.put("birchforest", 27);
        biome_names.put("birchforesthills", 28);
        biome_names.put("cold_ocean", 46);
        biome_names.put("coldbeach", 26);
        biome_names.put("colddeepocean", 49);
        biome_names.put("coldocean", 46);
        biome_names.put("coldtaiga", 30);
        biome_names.put("coldtaigahills", 31);
        biome_names.put("dark_forest", 29);
        biome_names.put("dark_forest_hills", 157);
        biome_names.put("deep_cold_ocean", 49);
        biome_names.put("deep_frozen_ocean", 50);
        biome_names.put("deep_lukewarm_ocean", 48);
        biome_names.put("deep_ocean", 24);
        biome_names.put("deep_warm_ocean", 47);
        biome_names.put("deepocean", 24);
        biome_names.put("desert", 2);
        biome_names.put("desert_hills", 17);
        biome_names.put("desert_lakes", 130);
        biome_names.put("deserthills", 17);
        biome_names.put("end_barrens", 43);
        biome_names.put("end_highlands", 42);
        biome_names.put("end_midlands", 41);
        biome_names.put("eroded_badlands", 165);
        biome_names.put("extremehills", 3);
        biome_names.put("extremehillsedge", 20);
        biome_names.put("extremehillsplus", 34);
        biome_names.put("flower_forest", 132);
        biome_names.put("forest", 4);
        biome_names.put("foresthills", 18);
        biome_names.put("frozen_ocean", 10);
        biome_names.put("frozen_river", 11);
        biome_names.put("frozendeepocean", 50);
        biome_names.put("frozenocean", 10);
        biome_names.put("frozenriver", 11);
        biome_names.put("giant_spruce_taiga", 160);
        biome_names.put("giant_spruce_taiga_hills", 161);
        biome_names.put("giant_tree_taiga", 32);
        biome_names.put("giant_tree_taiga_hills", 33);
        biome_names.put("gravelly_mountains", 131);
        biome_names.put("hell", 8);
        biome_names.put("ice_spikes", 140);
        biome_names.put("icemountains", 13);
        biome_names.put("iceplains", 12);
        biome_names.put("jungle", 21);
        biome_names.put("jungle_edge", 23);
        biome_names.put("jungle_hills", 22);
        biome_names.put("jungleedge", 23);
        biome_names.put("junglehills", 22);
        biome_names.put("lukewarm_ocean", 45);
        biome_names.put("lukewarmdeepocean", 48);
        biome_names.put("lukewarmocean", 45);
        biome_names.put("megataiga", 32);
        biome_names.put("megataigahills", 33);
        biome_names.put("mesa", 37);
        biome_names.put("mesaplateau", 39);
        biome_names.put("mesaplateau_f", 38);
        biome_names.put("modified_badlands_plateau", 167);
        biome_names.put("modified_gravelly_mountains", 162);
        biome_names.put("modified_jungle", 149);
        biome_names.put("modified_jungle_edge", 151);
        biome_names.put("modified_wooded_badlands_plateau", 166);
        biome_names.put("mountain_edge", 20);
        biome_names.put("mountains", 3);
        biome_names.put("mushroom_field_shore", 15);
        biome_names.put("mushroom_fields", 14);
        biome_names.put("mushroomisland", 14);
        biome_names.put("mushroomislandshore", 15);
        biome_names.put("nether", 8);
        biome_names.put("ocean", 0);
        biome_names.put("plains", 1);
        biome_names.put("river", 7);
        biome_names.put("roofedforest", 29);
        biome_names.put("savanna", 35);
        biome_names.put("savanna_plateau", 36);
        biome_names.put("savannaplateau", 36);
        biome_names.put("shattered_savanna", 163);
        biome_names.put("shattered_savanna_plateau", 164);
        biome_names.put("sky", 9);
        biome_names.put("small_end_islands", 40);
        biome_names.put("snowy_beach", 26);
        biome_names.put("snowy_mountains", 13);
        biome_names.put("snowy_taiga", 30);
        biome_names.put("snowy_taiga_hills", 31);
        biome_names.put("snowy_taiga_mountains", 158);
        biome_names.put("snowy_tundra", 12);
        biome_names.put("stone_shore", 25);
        biome_names.put("stonebeach", 25);
        biome_names.put("sunflower_plains", 129);
        biome_names.put("swamp", 6);
        biome_names.put("swamp_hills", 134);
        biome_names.put("swampland", 6);
        biome_names.put("taiga", 5);
        biome_names.put("taiga_hills", 19);
        biome_names.put("taiga_mountains", 133);
        biome_names.put("taigahills", 19);
        biome_names.put("tall_birch_forest", 155);
        biome_names.put("tall_birch_hills", 156);
        biome_names.put("the_end", 9);
        biome_names.put("the_void", 127);
        biome_names.put("warm_ocean", 44);
        biome_names.put("warmdeepocean", 47);
        biome_names.put("warmocean", 44);
        biome_names.put("wooded_badlands_plateau", 38);
        biome_names.put("wooded_hills", 18);
        biome_names.put("wooded_mountains", 34);

        probability.put(0, 0.0823387146);
        probability.put(1, 0.0857795715);
        probability.put(2, 0.0451049805);
        probability.put(3, 0.0640907288);
        probability.put(4, 0.0823188782);
        probability.put(5, 0.0411720276);
        probability.put(6, 0.0333244324);
        probability.put(7, 0.0367095947);
        probability.put(10, 0.0055770874);
        probability.put(11, 0.0014671326);
        probability.put(12, 0.0271652222);
        probability.put(13, 0.0086425781);
        probability.put(14, 0.0002738953);
        probability.put(15, 0.0002296448);
        probability.put(16, 0.0311912537);
        probability.put(17, 0.012223053);
        probability.put(18, 0.0214996338);
        probability.put(19, 0.0091415405);
        probability.put(21, 0.0103004456);
        probability.put(22, 0.0036964417);
        probability.put(23, 0.0007583618);
        probability.put(24, 0.0818778992);
        probability.put(25, 0.0042602539);
        probability.put(26, 0.0023468018);
        probability.put(27, 0.0239944458);
        probability.put(28, 0.0068023682);
        probability.put(29, 0.0238525391);
        probability.put(30, 0.0071723938);
        probability.put(31, 0.0017082214);
        probability.put(32, 0.0076217651);
        probability.put(33, 0.0033340454);
        probability.put(34, 0.0147125244);
        probability.put(35, 0.0267868042);
        probability.put(36, 0.0067504883);
        probability.put(37, 0.0043487549);
        probability.put(38, 0.0027374268);
        probability.put(39, 0.0011871338);
        probability.put(44, 0.0135391235);
        probability.put(45, 0.0361045837);
        probability.put(46, 0.0361434937);
        probability.put(48, 0.0270385742);
        probability.put(49, 0.0271568298);
        probability.put(50, 0.0079841614);
        probability.put(129, 0.0054008484);
        probability.put(130, 0.0020309448);
        probability.put(131, 0.0029457092);
        probability.put(132, 0.0045021057);
        probability.put(133, 0.0018669128);
        probability.put(134, 0.0010818481);
        probability.put(140, 0.0012680054);
        probability.put(149, 0.0004974365);
        probability.put(151, 4.5776e-06);
        probability.put(155, 0.0011688232);
        probability.put(156, 0.0008918762);
        probability.put(157, 0.0011436462);
        probability.put(158, 0.0003128052);
        probability.put(160, 0.0004119873);
        probability.put(161, 0.0003707886);
        probability.put(162, 0.0020050049);
        probability.put(163, 0.0012084961);
        probability.put(164, 0.0009040833);
        probability.put(165, 0.0002632141);
        probability.put(166, 0.0001373291);
        probability.put(167, 6.48499e-05);
        probability.put(168, 0.0007781982);
        probability.put(169, 0.0002746582);
    }

    public static WorldType wt;
    public static String generatorOptions;

    static class Test implements Comparable<Test> {
        int biome, X, Z;
        Test(int biome, int X, int Z) {
            this.biome = biome;
            this.X = X;
            this.Z = Z;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Test test = (Test) o;
            return biome == test.biome && X == test.X && Z == test.Z;
        }

        @Override
        public int hashCode() {
            return Objects.hash(biome, X, Z);
        }

        @Override
        public int compareTo(Test test) {
            if (this.equals(test)) return 0;
            if (this.biome != test.biome)
                return (int) Math.signum(probability.get(this.biome) - probability.get(test.biome));
            if (this.X != test.X) return this.X - test.X;
            return this.Z - test.Z;
        }
    }

    public static SortedSet<Test> tests = new TreeSet<Test>();

    static class SeedTester implements Runnable {
        final MinecraftInterface mc;
        final long lowbits;
        final AtomicInteger highbits;

        SeedTester(MinecraftInterface mc, long lowbits, AtomicInteger highbits) {
            this.mc = mc;
            this.lowbits = lowbits;
            this.highbits = highbits;
        }

        @Override
        public void run() {
            next_seed: while (true) {
                long highbits = this.highbits.getAndIncrement();
                if (highbits >= 0x10000) return;
                long seed = lowbits | (highbits << 48);
                try {
                    mc.createWorld(seed, wt, generatorOptions);
                    for (Test t: BiomeSeedTest.tests)
                        if (mc.getBiomeData(t.X, t.Z, 1, 1, false)[0] != t.biome)
                            continue next_seed;
                    synchronized (System.out) {
                        System.out.println(seed);
                    }
                } catch (MinecraftInterfaceException e) {
                    synchronized (System.err) {
                        System.err.println("Cannot get biome data: " + e.getLocalizedMessage());
                    }
                    return;
                }
            }
        }
    }

    static boolean nolog = false;
    public static void main(String[] args) {
        Options options = new Options();
        options.addOption("p", "mcpath", true, "Minecraft installatin path (.minecraft)");
        options.addRequiredOption("m", "mcversion", true, "Minecraft version (as displayed by the Minecraft launcher)");
        options.addOption("v", "verbose", false, "print extra informations");
        options.addOption("t", "worldtype", true, "world type: default, flat, largeBiomes, amplified, customized");
        options.addOption("g", "generator-options", true, "world generator options (format as in Minecraft save files)");
        CommandLine cmdline = null;
        try {
            cmdline = new DefaultParser().parse(options, args);
        } catch (ParseException e) {
            System.err.println("Bad argument: " + e.getLocalizedMessage());
            System.exit(1);
        }
        String installation_path = cmdline.getOptionValue('p', "~/.minecraft");
        String version = cmdline.getOptionValue('m');
        AmidstLogger.removeListener("master");
        AmidstLogger.removeListener("console");
        if (cmdline.hasOption('v'))
            AmidstLogger.addListener("BiomeSeedTest", (tag, msg) -> {
                if (BiomeSeedTest.nolog) return;
                synchronized (System.err) {
                    System.err.println("[" + tag + "] " + msg);
                }
            });
        wt = WorldType.from(cmdline.getOptionValue('t', "default"));
        generatorOptions = cmdline.getOptionValue('g', "");
        Pattern biome_regex = Pattern.compile("([\\-\\+]?\\d+)\\:([\\-\\+]?\\d+):([a-zA-Z_]+)");
        try {
            for (String test: cmdline.getArgList()) {
                Matcher m = biome_regex.matcher(test);
                if (!m.matches())
                    throw new IllegalArgumentException("invalid test specification " + test);
                String biomestr = m.group(3);
                if (!biome_names.containsKey(biomestr.toLowerCase()))
                    throw new IllegalArgumentException("unknown biome " + biomestr);
                tests.add(new Test(biome_names.get(biomestr.toLowerCase()), Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2))));
            }
        } catch (Exception e) {
            System.err.println("Cannot parse command line: " + e.getLocalizedMessage());
            System.exit(1);
        }

        int threads = Runtime.getRuntime().availableProcessors();
        MinecraftInterface[] interfaces = new MinecraftInterface[threads];
        try {
            MinecraftInstallation inst = MinecraftInstallation.newLocalMinecraftInstallation(installation_path);
            for (int t = 0; t < threads; t++) {
                if (t > 0) {
                    AmidstLogger.info("Silencing Minecraft interface initialization for additional threads.");
                    nolog = true;
                }
                interfaces[t] = MinecraftInterfaces.fromLocalProfile(inst.newLauncherProfile(version));
            }
            nolog = false;
        } catch (DotMinecraftDirectoryNotFoundException e) {
            System.err.println("Could not find minecraft installation at " + installation_path);
            System.exit(1);
        } catch (FormatException e) {
            System.err.println("Invalid version " + version);
            System.exit(1);
        } catch (IOException e) {
            System.err.println("Error reading from Minecraft installation: " + e.getLocalizedMessage());
            System.exit(1);
        } catch (MinecraftInterfaceCreationException e) {
            System.err.println("Cannot load Minecraft: " + e.getLocalizedMessage());
            System.exit(1);
        }

        ThreadPoolExecutor threadpool = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads);
        Scanner cin = new Scanner(System.in);
        while (cin.hasNextLine()) {
            long lowbits = Long.parseLong(cin.nextLine());
            if ((lowbits >>> 48) != 0) throw new IllegalArgumentException("Invalid input " + lowbits);
            AtomicInteger highbits = new AtomicInteger(0);
            for (MinecraftInterface mcintf: interfaces)
                threadpool.execute(new SeedTester(mcintf, lowbits, highbits));
        }
    }
}
