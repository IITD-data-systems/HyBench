#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdint>  // For uint64_t
#include <bits/stdc++.h>
using namespace std;

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
    
    int cou=0;
    while (std::getline(csvFile, line)) {
        cou+=1;
        if(cou%10000==0)
            cout<<cou<<endl;
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

int main(int argc, char* argv[]) {
    if(argc<3){
        cout<<"Usage: file_name offset_file_name"<<endl;
        return 0;
    }
    computeOffsets(argv[1], argv[2]);
    return 0;
}
