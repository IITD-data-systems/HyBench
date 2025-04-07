#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

// Function to read embeddings from CSV
vector<vector<float>> read_embeddings(const string &filename, int &dim) {
    ifstream file(filename);
    string line;
    vector<vector<float>> embeddings;

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<float> embedding;

        while (getline(ss, value, ',')) {
            embedding.push_back(stof(value)); // Convert string to float
        }

        if (embedding.empty()) continue;

        embeddings.push_back(embedding);
    }

    file.close();

    if (!embeddings.empty()) {
        dim = embeddings[0].size(); // Assume all embeddings have the same dimension
    }

    return embeddings;
}

int main() {
    string filename = "embeddings.csv";
    int dim = 0;
    
    // Read embeddings
    vector<vector<float>> embeddings = read_embeddings(filename, dim);
    int num_elements = embeddings.size();

    if (num_elements == 0) {
        cerr << "Error: No embeddings found in " << filename << endl;
        return 1;
    }

    // Convert to FAISS format
    float *data = new float[num_elements * dim];
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = embeddings[i][j];
        }
    }

    // Create HNSW index
    int M = 16;  // Number of neighbors per node
    faiss::IndexHNSWFlat index(dim, M, faiss::METRIC_L2);
    

    // Add embeddings to index
    index.add(num_elements, data);

    // Save index to file
    faiss::write_index(&index, "faiss_hnsw.index");

    cout << "FAISS HNSW index created and saved with " << num_elements << " embeddings." << endl;

    // Cleanup
    delete[] data;

    return 0;
}
