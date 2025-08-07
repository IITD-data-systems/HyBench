#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <random>
#include <cmath>
#include <faiss/IndexHNSW.h>  
#include <chrono>
#include <ctime>
#include <iomanip>

#define int long 
using namespace std;


int limit = 10000031;
int num_elements;
int dim = 384;
vector<vector<float>> embeddings;
// No normalization for L2

string filename;

void l2_normalize(vector<float>& vec) {
    float norm = 0.0f;
    for (float val : vec) norm += val * val;
    norm = std::sqrt(norm);
    
    if (norm > 0.0f) {
        for (float& val : vec) val /= norm;
    }
}

void read_embeddings(const string& filename, int &dim,int limit,faiss::IndexIVFFlat& index) {
    ifstream file(filename);
    string line;

    // int train_size = limit/100;
    int train_size= 316200;

    vector<float> train_data;
    train_data.reserve(train_size * dim);

    int cou = 0;
    int tot=0;
    while (getline(file, line)) {
        if (tot == train_size) break;
        if (cou % 100000 == 0) cout << cou << " read" << endl;
        cou++;
        

        // Trim outer quotes and square brackets
        if (line.front() == '"') line.erase(0, 1);
        if (line.back() == '"') line.pop_back();
        if (line.front() == '[') line.erase(0, 1);
        if (line.back() == ']') line.pop_back();

        stringstream ss(line);
        vector<float> embedding;
        string value;

        while (getline(ss, value, ',')) {
            embedding.push_back(stof(value));
        }
        if (index.metric_type == faiss::METRIC_INNER_PRODUCT) {
            l2_normalize(embedding);  // normalize to unit length for cosine similarity
        }

        if (!embedding.empty()) {
              
            train_data.insert(train_data.end(), embedding.begin(), embedding.end());
            tot+=1;
        }
    }

    // Train the index
    index.train(train_size, train_data.data());

    
}


void read_embeddings2(const string& filename, int &dim, int limit, faiss::IndexIVFFlat& index) {
    ifstream file(filename);
    string line;

    vector<float> all_embeddings;  // Flat array to store all embeddings
    all_embeddings.reserve(limit * dim);
    int count = 0;

    while (getline(file, line)) {
        if (count == limit) break;
        if (count%100000==0)cout<<count<<endl;

        if (line.front() == '"') line.erase(0, 1);
        if (line.back() == '"') line.pop_back();
        if (line.front() == '[') line.erase(0, 1);
        if (line.back() == ']') line.pop_back();

        stringstream ss(line);
        vector<float> embedding;
        string value;

        while (getline(ss, value, ',')) {
            embedding.push_back(stof(value));
        }

        if (!embedding.empty()) {
            if (index.metric_type == faiss::METRIC_INNER_PRODUCT) {
                l2_normalize(embedding);  // normalize to unit length for cosine similarity
            }
            if (dim == 0) dim = embedding.size();
            all_embeddings.insert(all_embeddings.end(), embedding.begin(), embedding.end());
            count++;
        }
    }

    if (count > 0) {
        index.add(count, all_embeddings.data());
    }
}





int fun1(string table, string metric) {
    string filename = "../data_csv_files/" + table + "_csv_files/embedding.csv";

    num_elements = limit;
    if (num_elements == 0) {
        cerr << "Error: No embeddings found in " << filename << endl;
        return 1;
    }

    int nlist = 3162;
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim); // safe for both L2 and IP

    faiss::MetricType faiss_metric;
    if (metric == "l2") {
        faiss_metric = faiss::METRIC_L2;
    } else if (metric == "cos") {
        faiss_metric = faiss::METRIC_INNER_PRODUCT;
    } else {
        cerr << "Error: Unsupported metric type '" << metric << "'. Use 'l2' or 'cos'." << endl;
        return 1;
    }

    faiss::IndexIVFFlat index(quantizer, dim, nlist, faiss_metric);

    
    read_embeddings(filename, dim, limit, index);
    read_embeddings2(filename, dim, limit, index);
    

    // Save index
    string index_filename = table + "_faiss_ivfflat_" + metric + "_final.index";
    faiss::write_index(&index, index_filename.c_str());

    cout << "FAISS IVFFlat (" << metric << ") index created and saved for "
         << table << " with " << num_elements << " embeddings." << endl;

    return 0;
}





signed main(int argc, char* argv[]) {
    if(argc<3){
        cout<< "first argument is either page or text, second argument is l2 or cos"<<endl;
        return 0;
    }
    ifstream infile("../dim");
	infile >> dim;
    
    auto start = std::chrono::high_resolution_clock::now();
    fun1(argv[1],argv[2]);    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Total execution time"<< duration.count() << " seconds\n";

    return 0;
}
