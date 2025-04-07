#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

pair<vector<faiss::idx_t>, vector<float>> KNNWithIndicesAndDistances(faiss::IndexHNSWFlat &index, float* query, int k) {
    vector<faiss::idx_t> indices(k);
    vector<float> distances(k);
    index.search(1, query, k, distances.data(), indices.data());
    return {indices, distances};
}

// Function to perform k-NN search using FAISS HNSW index
// **KNN Function to Return Only Indices**
vector<int> KNNWithIndicesOnly(faiss::IndexHNSWFlat &index, float* query, int k) {
    vector<int> indices(k);
    vector<float> distances(k);
    index.search(1, query, k, distances.data(), indices.data());
    return indices;
}



vector<float> KNNWithDistancesOnly(faiss::IndexHNSWFlat &index, float* query_data, int k) {
    vector<float> distances(k);
    vector<faiss::idx_t> dummy_indices(k);  // We ignore indices for now

    index.search(1, query_data, k, distances.data(), dummy_indices.data());

    return distances;
}


std::string getRowByIndex(const std::string& csvFilename, const std::string& offsetFilename, uint64_t rowIndex) {
    std::ifstream csvFile(csvFilename, std::ios::binary);
    std::ifstream offsetFile(offsetFilename, std::ios::binary);

    if (!csvFile.is_open() || !offsetFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return "";
    }

    // Seek to the correct offset in the binary file
    offsetFile.seekg(rowIndex * sizeof(uint64_t), std::ios::beg);
    uint64_t offset;
    offsetFile.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    if (offsetFile.gcount() != sizeof(uint64_t)) {
        std::cerr << "Invalid row index!" << std::endl;
        return "";
    }

    // Seek to the row in CSV and read the line
    csvFile.seekg(offset, std::ios::beg);
    std::string row;
    std::getline(csvFile, row);

    csvFile.close();
    offsetFile.close();

    return row;
}

std::vector<std::string> extractColumns(const std::string& row, const std::vector<int>& columnIndices) {
    std::vector<std::string> extractedColumns;
    std::stringstream ss(row);
    std::string cell;
    std::vector<std::string> allColumns;
    
    // Read CSV row while handling quotes correctly
    bool inQuotes = false;
    std::string token;

    while (std::getline(ss, cell, ',')) {
        if (!inQuotes) {
            if (!cell.empty() && cell.front() == '"') {
                inQuotes = true;
                token = cell;  // Start collecting quoted value
            } else {
                allColumns.push_back(cell);  // Normal unquoted value
            }
        } else {
            token += "," + cell;  // Append to quoted token
            if (!cell.empty() && cell.back() == '"') {
                inQuotes = false;
                allColumns.push_back(token);
            }
        }
    }

    // Extract required columns
    for (int colIndex : columnIndices) {
        if (colIndex < allColumns.size()) {
            extractedColumns.push_back(allColumns[colIndex]);
        } else {
            extractedColumns.push_back(""); // Handle missing columns
        }
    }

    return extractedColumns;
}



