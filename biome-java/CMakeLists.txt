cmake_minimum_required(VERSION 3.9)
project(minecrack-biome-java LANGUAGES NONE)

add_custom_command(OUTPUT minecraft-biome-java.jar
        COMMAND mvn package -Dbuilddir=${CMAKE_BINARY_DIR}
        MAIN_DEPENDENCY src/main/java/com/pisto/BiomeSeedTest.java
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        BYPRODUCTS original-minecrack-biome-java.jar maven-archiver maven-status generated-sources classes)
add_custom_target(minecraft-biome-java ALL DEPENDS minecraft-biome-java.jar)
