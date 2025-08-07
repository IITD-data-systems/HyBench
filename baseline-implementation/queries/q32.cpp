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
namespace q32{
string query_number="32";
queue<pair<int, float*>> queryQueue;
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



void runFullPipeline(const string& queryFilename,
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
    string csvFilenameExtra = "../../data_csv_files/page_csv_files/page_extra.csv";
    string offsetFilenameExtra = "../../offsets_files/page_extra_offsets.bin";
    ifstream csvFileExtra(csvFilenameExtra, std::ios::binary);
    ifstream offsetFileExtra(offsetFilenameExtra, std::ios::binary);

    if (!queryFile.is_open() || !csvFile.is_open() || !csvOffsetFile.is_open() ||
        !revFile.is_open() || !revOffsetFile.is_open() || !outFile.is_open()) {
        cerr << "Error: Could not open one or more files!" << endl;
        return;
    }

    unordered_map<int, int> actorCounts;

    string line;
    while (getline(queryFile, line)) {
        stringstream ss(line);
        int num_ranks;
        ss>>num_ranks;
        vector<int> ranks_list;
        int rank_max=0;
        for(int _=0;_<num_ranks;_++){
            int rank;
            ss>>rank;
            rank_max=max(rank_max,rank);
            ranks_list.push_back(rank);
        }
        int page_len;
        ss >> page_len;

        vector<float> query_embedding;
        float val;
        while (ss >> val) query_embedding.push_back(val);

        if (query_embedding.empty()) continue;

        float* query_data = new float[query_embedding.size()];
        copy(query_embedding.begin(), query_embedding.end(), query_data);

        // Run kNN
        
        int k_temp=3*rank_max;
        vector<long int> indices;
        while(true){
            indices = index.KNNWithIndicesOnly(query_data, k_temp);
            int count=0;
            vector<long int> final_indices;
            for(auto x : indices){
                if(x==-1)continue;
                int len = get_page_len(x,csvFileExtra,offsetFileExtra);
                if (len<page_len){
                    count++;
                    final_indices.push_back(x);
                }
                if (count==rank_max){
                    indices=final_indices;
                    break;
                }
            }
            if(count==rank_max)break;
            k_temp*=2;
        }
        delete[] query_data;

        for (auto x : ranks_list) {
            int rowId=indices[x-1];
            if (rowId == -1) continue;

            // Step 1: Extract row from CSV
            string row = getRowByIndex(csvFile, csvOffsetFile, rowId);

            // Step 2: Extract page ID
            int pageId = stoi(extractColumns(row, {0})[0]);

            // Step 3: Find revision rows
            auto it = revPageIndex.find(pageId);
            if (it == revPageIndex.end()) continue;
            const vector<int>& revOffsets = it->second;

            for (int index : revOffsets) {
                uint64_t offset;
                revOffsetFile.seekg(index * sizeof(uint64_t));
                revOffsetFile.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
                revFile.seekg(offset);
                string revRow;
                getline(revFile, revRow);

                // Step 4: Extract actor ID (column 3)
                string actorIdStr = extractColumns(revRow, {3})[0];
                int actorId = stoi(actorIdStr);

                actorCounts[actorId]++;
            }
        }
    }

    // Sort and write output
    vector<pair<int, int>> sortedActors(actorCounts.begin(), actorCounts.end());
    sort(sortedActors.begin(), sortedActors.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    for (const auto& [actorId, count] : sortedActors) {
        outFile << actorId << "," << count << "\n";
    }

    // Cleanup
    queryFile.close();
    csvFile.close();
    csvOffsetFile.close();
    revFile.close();
    revOffsetFile.close();
    outFile.close();
    csvFileExtra.close();
    offsetFileExtra.close();
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
    

    runFullPipeline(queryFilename,csvFilename,offsetFilename,revisionFilename,revisionOffsetFilename,revPageIndex,outputFilename,*index);
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
