# minecrack

minecrack can recover the full 64 bits seed of a Minecraft map from some of the map features.

minecrack is the first program to crack Minecraft seeds that recovers the full 64 bits seed without relying on assumptions on how the seed was generated. As long as the data you provide is correct and sufficient, any seed can be retrieved. The program is fully multithreaded.

With the current implementation, the seed is recovered in two steps:

* `minecrack-slime`: 48 bits are obtained from a list of [slime chunk](https://minecraft.gamepedia.com/Slime#.22Slime_chunks.22) coordinates
* `minecrack-biome`: the 48 bits candidates are tested for the remaining 16 bits by giving known biome coordinates.

The biome check has two implementations:

* `minecrack-biome`: a C++ version which uses the [cubiomes](https://github.com/Cubitect/cubiomes) library, a reimplementation in C of the Minecraft default world type generation
* `minecrack-biome-java.jar`: a Java version that calls directly the original Minecraft world generation code (through [Amidst](https://github.com/toolbox4minecraft/amidst))

The C version is obviously very fast, but it is the result of a reverse-engineering, and support for future versions of Minecraft (>= 1.15) is uncertain. The Java version is reasonably fast but ensures maximum compatibility, and it works with any world type (default, amplified, custom...).

# Building

Install JDK version >= 8, Maven, Boost.Program_options, then run:

```
git clone --recurse https://github.com/pisto/minecrack.git
mkdir minecrack/build
cd minecrack/build
cmake -DCMAKE_BUILD_TYPE=Release ..
#export JAVA_HOME=...
make
```

# Crack a seed

First collect around 18 slime chunk coordinates (chunk coordinates are world coordinates divided by 16 and rounded towards 0, the coordinates can be obtained from the debug screen). Then invoke `minecrack-slime` and append the chunk coordinates in `X:Z` format:

```
minecrack-slime -- 0:-96 0:-93 0:-77 0:-74 0:-73 0:-64 0:-55 0:-51 0:-50 0:-49 0:-46 0:-25 0:-4 0:11 0:30 0:37 0:44 > 48bit
```

The output file `48bit` will contain all possible 48 bits map seeds that can generate the specified slime chunks.

These results are then passed to either `minecrack-biome` or `minecrack-biome-java.jar`, which output the final candidate seed(s). In both cases you need to provide a number of biome "samplings": just walk around in the map and every 100 blocks or so write down the block coordinate and the biome name, as shown in the debug screen. Then, invoke one of the two implementations.

## `minecrack-biome`

Format the biome samplings as `X:Z:biome_name`, and append them to the invokation:

```
minecrack-biome -m 1.14 -- -212:-8:mushroom_fields 1044:-776:mountains -1208:-1004:swamp -2408:-796:desert_hills -5622:-1596:river -4608:4267:frozen_ocean < 48bit
```

The Minecraft version is specified with `-m <version>`. Only use the major version number (e.g. `1.14` even if the server runs 1.14.4). The supported versions are from `1.7` through `1.14`, and `BE` for the Bedrock Edition.


## `minecrack-biome-java`

First make sure that you have the desired specific version of Minecraft installed. You can do so by creating a new Minecraft profile in the Minecraft launcher, and start the game once. Then, format the biome samplings just as for `minecrack-biome`, and invoke `minecrack-biome-java.jar`:

```
java -jar minecrack-biome-java.jar -m 1.14.4 -- -212:-8:mushroom_fields 1044:-776:mountains -1208:-1004:swamp -2408:-796:desert_hills -5622:-1596:river -4608:4267:frozen_ocean < 48bit
```

The Minecraft version is specified with `-m <version>` and it has to be the same string as shown in the Minecraft launcher (like `1.13.2` or `19w34a`). The `-t <type>` argument specifies the world type (`default`, `flat`, `largeBiomes`, `amplified`, `customized`). The generator options can be set with the argument `-g <string>`, but use this parameter only if you know what you are doing. The Java version will use [Amidst](https://github.com/toolbox4minecraft/amidst) internally to locate your Minecraft installation (`.minecraft`), attach to it and run the tests. If your Minecraft installation is in a non-standard path, pass the argument `--mcpath <path_to_.minecraft>`.

# Technical details

To come later.
