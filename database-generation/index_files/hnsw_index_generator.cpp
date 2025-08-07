#include "hnswlib/hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#define int long
using namespace std;

int limit = 10000031;
int dim = 384;

// Normalize vector to unit length for cosine similarity
void l2_normalize(vector<float>& vec) {
    float norm = 0.0f;
    for (float val : vec) norm += val * val;
    norm = sqrt(norm);
    if (norm > 0.0f) {
        for (float& val : vec) val /= norm;
    }
}

void read_embeddings_to_add(const string& filename, hnswlib::HierarchicalNSW<float>* index, bool normalize) {
    ifstream file(filename);
    string line;
    int count = 0;

    vector<float> embedding(dim);

    while (getline(file, line)) {
        if (count == limit) break;

        if (!line.empty() && line.front() == '"') line.erase(0, 1);
        if (!line.empty() && line.back() == '"') line.pop_back();
        if (!line.empty() && line.front() == '[') line.erase(0, 1);
        if (!line.empty() && line.back() == ']') line.pop_back();

        stringstream ss(line);
        string value;
        int i = 0;
        while (getline(ss, value, ',') && i < dim) {
            embedding[i++] = stof(value);
        }

        if (i == dim) {
            if (normalize) l2_normalize(embedding);
            index->addPoint(embedding.data(), count);
            count++;
            if (count % 1000 == 0) {
                cout << count << " points added." << endl;
            }
        } else {
            cerr << "Warning: embedding dimension mismatch at line " << count + 1 << endl;
        }
    }

    cout << "Total embeddings added: " << count << endl;
}

int fun2(const string& table, const string& metric) {
    cout << "begin" << endl;
    string filename = "../data_csv_files/" + table + "_csv_files/embedding.csv";

    // === Choose space based on metric ===
    hnswlib::SpaceInterface<float>* space;
    bool normalize = false;

    if (metric == "l2") {
        space = new hnswlib::L2Space(dim);
    } else if (metric == "cos") {
        space = new hnswlib::InnerProductSpace(dim);
        normalize = true;
    } else {
        cerr << "Error: invalid metric '" << metric << "'. Use 'l2' or 'cos'." << endl;
        return 1;
    }

    int M = 16;
    int ef_construction = 64;
    hnswlib::HierarchicalNSW<float> index(space, limit, M, ef_construction);

    read_embeddings_to_add(filename, &index, normalize);

    string index_name = table + "_hnswlib_" + metric + "_final.index";
    index.saveIndex(index_name);

    cout << "hnswlib index created and saved as '" << index_name << "' with " << limit << " embeddings." << endl;

    delete space;
    return 0;
}

signed main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "first argument is table name (e.g., 'page' or 'text'), second is metric: 'l2' or 'cos'" << endl;
        return 0;
    }

    ifstream infile("../dim");
	infile >> dim;

    auto start = chrono::high_resolution_clock::now();
    fun2(argv[1], argv[2]);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Total execution time: " << duration.count() << " seconds\n";
    return 0;
}
