#include <bits/stdc++.h>
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

void saveVectorOfVectors(const vector<vector<float>>& data, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out) throw runtime_error("Cannot open file for writing.");

    size_t outerSize = data.size();
    out.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

    for (const auto& inner : data) {
        size_t innerSize = inner.size();
        out.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
        out.write(reinterpret_cast<const char*>(inner.data()), innerSize * sizeof(float));
    }

    out.close();
}

void read_embeddings(const string& table) {
    string filename = "../data_csv_files/" + table + "_csv_files/embedding.csv";
    ifstream file(filename);
    string line;

    vector<vector<float>> data;
    

    int cou = 0;
    int tot=0;
    while (getline(file, line)) {
        if (tot == limit) break;
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

        if (!embedding.empty()) {
              
            data.push_back(embedding);
            tot+=1;
        }
    }
    string outputFilename= table+"_embeddings.bin";
    saveVectorOfVectors(data,outputFilename);
  

    
}

signed main(int argc, char* argv[]) {
    if(argc<2){
        cout<< "first argument is either page or text"<<endl;
        return 0;
    }

    
    string table=argv[1];
    ifstream infile("dim");
	infile >> dim;

    
    auto start = std::chrono::high_resolution_clock::now();
    read_embeddings(table);    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Total execution time"<< duration.count() << " seconds\n";
    return 0;
}
