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
namespace q11{
string query_number="11";
queue<pair<int, float*>> queryQueue;
queue<int> rowIdQueue;
queue<string> rowQueue;
queue<int> pageIdQueue;
queue<string> actorQueue;
queue<pair<int,int>> actorEditCountQueue;
queue<vector<pair<int, int>>> outputQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxPageId, mtxActor, mtxActorEditCount, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvPageId, cvActor, cvActorEditCount, cvOutput;

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

void fullPipeline(const string& queryFilename,
                  const string& csvFilename,
                  const string& csvOffsetFilename,
                  const string& revisionsFilename,
                  const string& revOffsetFilename,
                  const unordered_map<int, vector<int>>& revPageIndex,
                  const string& outputFilename,
                  MyIndex& index)
{
    ifstream queryFile(queryFilename);
    ifstream csvFile(csvFilename, ios::binary);
    ifstream csvOffsetFile(csvOffsetFilename, ios::binary);
    ifstream revFile(revisionsFilename, ios::binary);
    ifstream revOffsetFile(revOffsetFilename, ios::binary);
    ofstream outFile(outputFilename);

    if (!queryFile.is_open()) {
        cerr << "Error: Could not open query file!" << endl;
        return;
    }

    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }

    unordered_map<int, int> actorCounts;  

    string line;
    while (getline(queryFile, line)) {
        stringstream ss(line);

        int k;
        ss >> k;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (query_embedding.empty()) continue;

        float* query_data = new float[query_embedding.size()];
        copy(query_embedding.begin(), query_embedding.end(), query_data);

        auto indices = index.KNNWithIndicesOnly(query_data, k);
        delete[] query_data;

        for (int rowId : indices) {
            if (rowId == -1) continue;

            // Get row and extract page ID
            string row = getRowByIndex(csvFile, csvOffsetFile, rowId);
            int pageId = stoi(extractColumns(row, {0})[0]);

            // Get revision rows for page ID
            auto revRows = getRowByPageID(revFile, revOffsetFile, pageId, revPageIndex);

            for (const auto& revRow : revRows) {
                auto cols = extractColumns(revRow, {3, 2});  // actor_id, minor_edit
                int actorId = stoi(cols[0]);
                int minor_edit = stoi(cols[1]);
                actorCounts[actorId] += minor_edit;
            }
        }
    }

    // Sort actors by edit count descending
    vector<pair<int, int>> sortedActors(actorCounts.begin(), actorCounts.end());
    sort(sortedActors.begin(), sortedActors.end(),
         [](const auto& a, const auto& b) {
             return a.second > b.second;
         });

    for (const auto& [actorId, count] : sortedActors) {
        outFile << actorId << "," << count << "\n";
    }

    queryFile.close();
    csvFile.close();
    csvOffsetFile.close();
    revFile.close();
    revOffsetFile.close();
    outFile.close();
}


void q(string query_size,MyIndex* index,unordered_map<int, std::vector<int>>& revPageIndex,int search_parameter) {
    
    string csvFilename = "../../database-generation/data_csv_files/text_csv_files/text.csv";
    string offsetFilename = "../../database-generation/offsets_files/text_offsets.bin";
    string revisionFilename = "../../database-generation/data_csv_files/revision_csv_files/revision_clean.csv";
    string revisionOffsetFilename = "../../database-generation/offsets_files/revision_offsets.bin";
    string revPageIndexFilename = "../../database-generation/index_files/rev_id_index.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    
    

    
    index->set_search_parameter(search_parameter);
    vector<int> columnIndices = {0, 1}; 
    

    auto start = std::chrono::high_resolution_clock::now();
    

    fullPipeline(queryFilename,csvFilename,offsetFilename,revisionFilename,revisionOffsetFilename,revPageIndex,outputFilename,*index);
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
