#include <bits/stdc++.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>   // Needed for quantizer
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "../pipeline_stages.cpp"
#include <unordered_map>
#include <chrono>

using namespace std;

// Shared Queues
namespace q14{
string query_number="14";
queue<pair<int,int>> yearRevIdQueue;


// Mutexes and Condition Variables
mutex mtxRevId;
condition_variable cvRevId;

bool no_more_queries1 = false;

map<int,set<pair<float,int>>> yearToDistancesWithOldId;

float d;
int start_year, end_year;
float* query_embedding;

int limit=10000031;
int dim=384;
vector<float> read_embedding(string& embedding_str) {
    vector<float> embedding;
    stringstream ss;

    // Remove surrounding quotes if present
    if (!embedding_str.empty() && embedding_str.front() == '"' && embedding_str.back() == '"') {
        embedding_str = embedding_str.substr(1, embedding_str.size() - 2);
    }

    // Remove brackets and push valid characters to stringstream
    for (char ch : embedding_str) {
        if (ch != '[' && ch != ']')
            ss << ch;
    }

    string value;
    while (getline(ss, value, ',')) {
        // Handle potential whitespace
        try {
            embedding.push_back(stof(value));
        } catch (...) {
            // Skip or handle parse error
        }
    }

    return embedding;
}


vector<vector<float>> read_embeddings(const string& filename, int &dim,int limit) {
    ifstream file(filename);
    string line;
    vector<vector<float>> embeddings;
    int cou = 0;

    while (getline(file, line)) {
        if (cou == limit) break;
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
            embeddings.push_back(embedding);  // no normalization
        }
    }

    if (!embeddings.empty()) {
        dim = embeddings[0].size();
    }

    return embeddings;
}


float get_L2_distance(float* query_embedding, vector<float>& embedding) {
    float sum = 0.0f;
    for (size_t i = 0; i < embedding.size(); ++i) {
        float diff = query_embedding[i] - embedding[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

float get_cos_distance(float* query_embedding, std::vector<float>& embedding) {
    float dot = 0.0f;
    float norm_query = 0.0f;
    float norm_embedding = 0.0f;

    for (size_t i = 0; i < embedding.size(); ++i) {
        dot += query_embedding[i] * embedding[i];
        norm_query += query_embedding[i] * query_embedding[i];
        norm_embedding += embedding[i] * embedding[i];
    }

    norm_query = std::sqrt(norm_query);
    norm_embedding = std::sqrt(norm_embedding);

    if (norm_query == 0 || norm_embedding == 0) {
        return 1.0f; // max cosine distance if one vector is zero
    }

    float cosine_similarity = dot / (norm_query * norm_embedding);
    float cosine_distance = 1.0f - cosine_similarity;
    return cosine_distance;
}




// Stage 1: Query Reader — updated to read `k` instead of `d`
void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        ss >> d >> start_year >> end_year;
        vector<float> q;
        float val; while (ss >> val) q.push_back(val);
        if (!q.empty()) {
            query_embedding = new float[q.size()];
            copy(q.begin(), q.end(), query_embedding);
        }
    }
    file.close();
    
}

// Stage 2: iKNN on each year index
void revIdCollectorThread(const string& revisionsFilename) {
    ifstream revFile(revisionsFilename, ios::binary);
    string line;
    

    while (getline(revFile, line)) {
        auto extracted= extractColumns(line, {0,4});
        string revIdStr = extracted[0];
        string timestamp = extracted[1];

        if (timestamp.length() >= 4) {
            int year = stoi(timestamp.substr(0, 4));
            if (year >= start_year && year <= end_year) {
                int revId = stoi(revIdStr);
                {
                
                yearRevIdQueue.push({year, revId});
                }
                

            }
        }
    }
    no_more_queries1=true;
    
    revFile.close();
    
}

void distanceCollectorThread(const unordered_map<int, vector<int>>& old_id_index, const string& embedding_offset_file, const string& embeddingFile, const string& outputFilename,vector<vector<float>>& embeddings,string metric_type){
    ifstream offset_file(embedding_offset_file, ios::binary);
    ifstream embedding_file(embeddingFile, ios::binary);;
    

    while(true){
        

        if (yearRevIdQueue.empty() && no_more_queries1) break;

        auto [year,revId] = yearRevIdQueue.front();
        yearRevIdQueue.pop();
        
        
        
        auto row=old_id_index.find(revId);
        if(row==old_id_index.end()){
            
            continue;
        }
        int rowIndex = row->second[0];
        
        if(rowIndex>=limit)
        continue;
        std::vector<float>& embedding=embeddings[rowIndex];
        
        float dist;
        if(metric_type=="l2"){
        dist= get_L2_distance(query_embedding, embedding);
        }
        else{
            dist= get_cos_distance(query_embedding, embedding);
        }
        if(dist<=d){
            yearToDistancesWithOldId[year].insert(make_pair(dist,revId)); 
        }
        
        
        

    }
    
    offset_file.close();
    embedding_file.close();
    ofstream outFile(outputFilename);
    for(auto& [x,y] : yearToDistancesWithOldId){
        for(auto& [dist, old_id] : y){
            outFile << x << " " << old_id << " " << dist << "\n";
        }
    }
    outFile.close();
    

}




void q(string query_size,unordered_map<int, vector<int>>& old_id_index,vector<vector<float>>& embeddings,string metric_type) {
    
    string revision_filename = "../../database-generation/data_csv_files/revision_csv_files/revision_clean.csv";
    string embedding_offset_file = "../../database-generation/offsets_files/text_embedding_offsets.bin";
    string embedding_file =  "../../database-generation/data_csv_files/text_csv_files/embedding.csv";
    string query_filename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + metric_type + "_" + query_size + ".txt";
    string output_filename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + metric_type + "_" + query_size + ".txt";
    string old_id_index_filename = "../../database-generation/index_files/old_id_index.bin";

    ifstream infile("../../database-generation/dim");
	infile >> dim;
    
    auto start = std::chrono::high_resolution_clock::now();

    

    queryReaderThread(query_filename);
    revIdCollectorThread(revision_filename);
    distanceCollectorThread(old_id_index, embedding_offset_file, embedding_file, output_filename, embeddings, metric_type);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time for query " + query_number + " subtype " + query_size + ": " << duration.count()*1000 << " seconds\n";

    ofstream time_file("../../output-files/queries_time",ios::app);
    time_file << "brute" << " " << metric_type << " " << query_number << " "<< query_size << " " << duration.count()*1000 <<endl;
    time_file.close();
    
    no_more_queries1 = false;
    // Clear the map
    yearToDistancesWithOldId.clear();

    // Empty the queue
    while (!yearRevIdQueue.empty()) {
        yearRevIdQueue.pop();
    }


    

    
}
}
