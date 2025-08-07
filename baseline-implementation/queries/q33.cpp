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

// Shared Queues for Each Stage
namespace q33{
string query_number="33";

queue<tuple<vector<pair<float,float>>,int, float*>> queryQueue;
queue<int> rowIdQueue;
queue<string> rowQueue;
queue<int> pageIdQueue;
queue<string> actorIdQueue;
queue<int> actorCountQueue;
queue<vector<pair<int, int>>> outputQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxPageId, mtxActorId, mtxActorCount, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvPageId, cvActorId, cvActorCount, cvOutput;

// Control Variables
bool no_more_queries1 = false;
bool no_more_queries2 = false;
bool no_more_queries3 = false;
bool no_more_queries4 = false;
bool no_more_queries5 = false;
bool no_more_queries6 = false;
bool no_more_queries7 = false;


// helper function
std::vector<std::string> getRowByPageID(ifstream& revFile, ifstream& offsetFile,
                                        int pageId,
                                        const std::unordered_map<int, std::vector<int>>& revPageIndex) {
    std::vector<std::string> result;

    auto it = revPageIndex.find(pageId);
    if (it == revPageIndex.end()) return result;

    const std::vector<int>& rowIndices = it->second;

    if (!offsetFile) throw std::runtime_error("Failed to open offset file");    
    if (!revFile) throw std::runtime_error("Failed to open revisions file");

    for (int rowId : rowIndices) {
        uint64_t offset;
        offsetFile.seekg(rowId * sizeof(uint64_t));
        offsetFile.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

        revFile.seekg(offset);
        std::string line;
        std::getline(revFile, line);
        result.push_back(std::move(line));
    }

    return result;
}


// pipeline_stages

// **Query Reader Thread**
void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    if (!file.is_open()) {
        cerr << "Error: Could not open query file!" << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int num_constraints;
        ss >> num_constraints;
        vector<pair<float,float>> distance_ranges;
        for(int _=0;_<num_constraints;_++){
            float d,d_star;
            ss >> d >> d_star;
            distance_ranges.push_back({d,d_star});
        }
        int page_len;
        ss >> page_len;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            
            queryQueue.push({distance_ranges, page_len,query_data});
        }
        
    }

    file.close();
    no_more_queries1 = true;
    
    
}

// **iKNN Thread**
void iKNNThread(MyIndex &index) {
    string csvFilename = "../../data_csv_files/page_csv_files/page_extra.csv";
    string offsetFilename = "../../offsets_files/page_extra_offsets.bin";
    ifstream csvFile(csvFilename, std::ios::binary);
    ifstream offsetFile(offsetFilename, std::ios::binary);
    while (true) {
        

        if (queryQueue.empty() && no_more_queries1) break;

        auto [distance_ranges,page_len, query_data] = queryQueue.front();
        queryQueue.pop();
        
        float mi=1000000;
        float ma=-1000000;
        for(auto& [x,y] : distance_ranges){
            mi=min(mi,x);
            ma=max(ma,y);
        }
        
        auto [indices,distances] = index.KNNWithDistanceUpDownWithDistances(query_data,mi,ma);
        
        for (int i = 0; i < indices.size(); i++) {
            if(indices[i]==-1)continue;
            if (!range_lie_identifier(distances[i],distance_ranges))continue;
            
            int len=get_page_len(indices[i],csvFile,offsetFile);
            if(len >= page_len)continue;
            rowIdQueue.push(indices[i]);
            
        }

        delete[] query_data;

        
    }
    no_more_queries2=true;
    
    csvFile.close();
    offsetFile.close();
    cout<<"done2";
    
    
}

// **Row Extractor Thread**
void rowExtractorThread(const string& csvFilename, const string& offsetFilename) {
    ifstream csvFile(csvFilename, std::ios::binary);
    ifstream offsetFile(offsetFilename, std::ios::binary);
    while (true) {
        

        if (rowIdQueue.empty() && no_more_queries2) break;

        int rowId = rowIdQueue.front();
        rowIdQueue.pop();
        

        
        rowQueue.push(getRowByIndex(csvFile, offsetFile, rowId));
        

    }
    
    csvFile.close();
    offsetFile.close();
    no_more_queries3=true;
    
}

// **Column Extractor Thread (Extract Page IDs)**
void columnExtractorThread(const vector<int>& columnIndices) {
    while (true) {
        

        if (rowQueue.empty() && no_more_queries3) break;

        string row = rowQueue.front();
        rowQueue.pop();
        
        pageIdQueue.push(stoi(extractColumns(row, columnIndices)[0]));
        
    }
    
    no_more_queries4=true;
    
}

// **Row Extractor Thread for Revisions**
void rowExtractorThread1(const std::string& revisionsFilename,
                         const std::string& offsetFilename,
                         const std::unordered_map<int, std::vector<int>>& revPageIndex) {
    ifstream revFile(revisionsFilename, std::ios::binary);
    ifstream offsetFile(offsetFilename, std::ios::binary);
    
    while (true) {
        

        if (pageIdQueue.empty() && no_more_queries4) break;

        int pageId = pageIdQueue.front();
        pageIdQueue.pop();
        
        
        vector<string> extracted = getRowByPageID(revFile, offsetFile, pageId, revPageIndex);
        
        for(auto row : extracted){
            
            actorIdQueue.push(row);
            
        }
        
        
    }
    revFile.close();
    offsetFile.close();
    no_more_queries5=true;
    
}

// **Column Extractor Thread (Extract Actor IDs)**
void columnExtractorThread1() {
    while (true) {
        

        if (actorIdQueue.empty() && no_more_queries5) break;

        string row = actorIdQueue.front();
        actorIdQueue.pop();
        
        actorCountQueue.push(stoi(extractColumns(row, {3})[0]));
        
    }
    
    no_more_queries6=true;
    
}

// **Actor Aggregator Thread**
void actorAggregatorThread() {
    
    unordered_map<int, int> actorCounts;
    while (true) {
        

        if (actorCountQueue.empty() && no_more_queries6) break;

        int actorId = actorCountQueue.front();
        actorCountQueue.pop();
        
        
        actorCounts[actorId]++;
        
    }
    vector<pair<int, int>> sortedActors(actorCounts.begin(), actorCounts.end());
    sort(sortedActors.begin(), sortedActors.end(), [](auto& a, auto& b) { return a.second > b.second; });

    
    outputQueue.push(sortedActors);
    
    no_more_queries7=true;
    
}

void outputWriterThread(const string& outputFilename) {
    ofstream file(outputFilename);
    if (!file.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }
    
    while (true) {
        
        
        if (outputQueue.empty() && no_more_queries7) break;
        
        vector<pair<int, int>> sortedActors = outputQueue.front();
        outputQueue.pop();
        
        
        for (const auto& [actorId, count] : sortedActors) {
            file << actorId << "," << count << "\n";
        }
        
    }
    
    file.close();

}



void q(string query_size, MyIndex* index, unordered_map<int, std::vector<int>>& revPageIndex,int search_parameter) {
    
    string csvFilename = "../../database-generation/data_csv_files/page_csv_files/page.csv";
    string offsetFilename = "../../database-generation/offsets_files/page_offsets.bin";
    string revisionFilename = "../../database-generation/data_csv_files/revision_csv_files/revision_clean.csv";
    string revisionOffsetFilename = "../../database-generation/offsets_files/revision_offsets.bin";
    string revPageIndexFilename = "../../database-generation/index_files/rev_page_index.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    
    
    
    index->set_search_parameter(search_parameter);
    vector<int> columnIndices = {0, 1};

    


    auto start = std::chrono::high_resolution_clock::now();
    

    queryReaderThread(queryFilename);
    iKNNThread(*index);
    rowExtractorThread(csvFilename, offsetFilename);
    columnExtractorThread(columnIndices);
    rowExtractorThread1(revisionFilename, revisionOffsetFilename, revPageIndex);
    columnExtractorThread1();
    actorAggregatorThread();
    outputWriterThread(outputFilename);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time for query " + query_number + " subtype " + query_size + ": " << duration.count()*1000 << " seconds\n";

    ofstream time_file("../../output-files/queries_time",ios::app);
    time_file << index->index_kind() << " " << index->metric_type() << " " << query_number << " "<< query_size << " " << duration.count() <<endl;
    time_file.close();
    
    no_more_queries1 = false;
    no_more_queries2 = false;
    no_more_queries3 = false;
    no_more_queries4 = false;
    no_more_queries5 = false;
    no_more_queries6 = false;
    no_more_queries7 = false;
    
}
}
