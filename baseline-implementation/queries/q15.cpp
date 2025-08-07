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
namespace q15{
string query_number="15";
queue<pair<string,int>> categoryPageIdQueue;
mutex mtxPageId;
condition_variable cvPageId;


int embedding_size=384;

bool no_more_queries1 = false;

int dim=384;
int limit=9997744;



vector<float> read_embedding(string& embedding_str) {
    vector<float> embedding;
    stringstream ss;

    // Remove brackets if present
    for (char ch : embedding_str) {
        if (ch != '[' && ch != ']')
            ss << ch;
    }

    string value;
    while (getline(ss, value, ',')) {
        embedding.push_back(stof(value));
    }

    return embedding;
}

float get_L2_distance(vector<float>& query_embedding, vector<float>& embedding) {
    float sum = 0.0f;
    for (size_t i = 0; i < embedding.size(); ++i) {
        float diff = query_embedding[i] - embedding[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


float get_cos_distance(vector<float>& query_embedding, vector<float>& embedding) {
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


map<string,vector<int>> category_page_id_map;
map<string,vector<float>> category_sum_embedding_map;
map<string,pair<float,int>> category_final_page_id_map;

void add_embeddings(vector<float>& sum_vector, vector<float>& single_vector){
    for(int i=0;i<single_vector.size();i++){
        sum_vector[i]+=single_vector[i];
    }
}


void categoryPageIdCollectorThread(const string& categoryFilename) {
    ifstream categoryFile(categoryFilename, ios::binary);
    string line;
    getline(categoryFile,line);
    while (getline(categoryFile, line)) {
        auto extracted= extractColumns(line, {0,1});
        int pageId = stoi(extracted[0]);
        string category_title = extracted[1];
        
        if(category_page_id_map.find(category_title)==category_page_id_map.end()){
            category_page_id_map[category_title]={};
            category_sum_embedding_map[category_title]=vector<float>(embedding_size, 0.0f);
            category_final_page_id_map[category_title]=make_pair((float)1e9,-1);
        }
        category_page_id_map[category_title].push_back(pageId);
        
        {
            
            categoryPageIdQueue.push({category_title, pageId});
        }
        
    }
    no_more_queries1=true;
    
    categoryFile.close();
}

void average_embedding_calculator_thread(const unordered_map<int, vector<int>>& page_id_index,vector<vector<float>>& embeddings){
    
    
    while(true){
        

        if (categoryPageIdQueue.empty() && no_more_queries1) break;

        auto [category_title,pageId] = categoryPageIdQueue.front();
        categoryPageIdQueue.pop();
        
        
        auto it = page_id_index.find(pageId);
        int index=(it->second)[0];
        
        vector<float> embedding= embeddings[index];
        add_embeddings(category_sum_embedding_map[category_title],embedding);
        
    }
    for(auto& [category_title,sum_embedding] : category_sum_embedding_map){
        float size=(float)category_page_id_map[category_title].size();
        for(int i=0;i<embedding_size;i++){
            sum_embedding[i]/=size;
        }
    }
    
}

void final_category_page_id_creator( const string& outputFilename,const unordered_map<int, vector<int>>& page_id_index,vector<vector<float>>& embeddings,string metric_type){
    
    for(auto [category_title,pageIds]: category_page_id_map){
        for(auto pageId : pageIds){
        auto it = page_id_index.find(pageId);
        int index=(it->second)[0];
        
        vector<float> embedding= embeddings[index];
        float dist;
        if(metric_type=="l2"){
        dist=get_L2_distance(embedding,category_sum_embedding_map[category_title]);
        }
        else{
        dist=get_cos_distance(embedding,category_sum_embedding_map[category_title]);
        }
        
        if(category_final_page_id_map[category_title].second==-1){
            category_final_page_id_map[category_title]=make_pair(dist,pageId);
            
        }
        else if(dist<category_final_page_id_map[category_title].first || (category_final_page_id_map[category_title].first==dist && category_final_page_id_map[category_title].second>pageId)){
            
            category_final_page_id_map[category_title]=make_pair(dist,pageId);
            
        }
        }
    }
    ofstream outFile(outputFilename);
    for(auto& [x,y] : category_final_page_id_map){
        outFile << x << " " << y.second << "\n";
    }


}




void q(unordered_map<int, vector<int>>& page_id_index,vector<vector<float>>& embeddings,string metric_type) {
    
    string categoryFilename = "../../database-generation/data_csv_files/category_csv_files/category_links_clean.csv";
    string page_embedding_file =  "../../database-generation/data_csv_files/page_csv_files/embedding.csv";
    string output_filename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + metric_type + ".txt";
    string page_id_index_filename = "../../database-generation/index_files/page_id_index.bin";
    

    auto start = std::chrono::high_resolution_clock::now();

    ifstream infile("../../database-generation/dim");
	infile >> dim;
    embedding_size=dim;
    

    categoryPageIdCollectorThread(categoryFilename);
    average_embedding_calculator_thread(page_id_index, embeddings);
    final_category_page_id_creator(output_filename, page_id_index, embeddings, metric_type);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time for query " + query_number + ": " << duration.count()*1000 << " seconds\n";

    ofstream time_file("../../output-files/queries_time",ios::app);
    time_file << "brute" << " " << metric_type << " " << query_number << " "<< 1 << " " << duration.count()*1000 <<endl;
    time_file.close();
    
    no_more_queries1 = false;
    

    
}
}
