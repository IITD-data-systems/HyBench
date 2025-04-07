#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdint>  // For uint64_t

void computeOffsets(const std::string& csvFilename, const std::string& offsetFilename) {
    std::ifstream csvFile(csvFilename, std::ios::binary);
    std::ofstream offsetFile(offsetFilename, std::ios::binary);

    if (!csvFile.is_open() || !offsetFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return;
    }

    std::vector<uint64_t> offsets;
    uint64_t currentOffset = 0;
    std::string line;

    while (std::getline(csvFile, line)) {
        offsets.push_back(currentOffset);
        currentOffset = csvFile.tellg();  // Get next line's start position
        if (currentOffset == static_cast<uint64_t>(-1)) break;  // End of file check
    }

    // Write offsets to binary file
    offsetFile.write(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(uint64_t));

    csvFile.close();
    offsetFile.close();

    std::cout << "Offsets stored in " << offsetFilename << std::endl;
}

int main() {
    computeOffsets("embeddings.csv", "offsets.bin");
    return 0;
}
