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
namespace q6{
string query_number="6";
queue<tuple<float, float*>> queryQueue;
queue<int> rowIdQueue;
queue<string> rowQueue;
queue<int> oldIdQueue;
queue<string> revisionRowQueue;
queue<pair<int, string>> revTimestampQueue;
queue<int> filteredRevIdQueue;


// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxOldId, mtxRevOffset, mtxRevRow, mtxRevTime, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvOldId, cvRevOffset, cvRevRow, cvRevTime, cvOutput;


bool no_more_queries1 = false;
bool no_more_queries2 = false;
bool no_more_queries3 = false;
bool no_more_queries4 = false;
bool no_more_queries5 = false;
bool no_more_queries6 = false;
bool no_more_queries7 = false;

string d1,d2;
int k;

vector<string> extractNumbersBeforeComma(const std::string& filePath) {
    std::vector<string> numbers;
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << "\n";
        return numbers;
    }
    int cou=0;

    while (std::getline(file, line)) {
        cou+=1;
        if(cou%100000==0)
        cout<<cou<<endl;
        std::istringstream ss(line);
        std::string token;
        if (std::getline(ss, token, ',')) {
            try {
                string value = token;
                numbers.push_back(value);
            } catch (const std::invalid_argument&) {
                // Skip if the token is not a valid number
                continue;
            }
        }
    }

    return numbers;
}

std::vector<std::string> getRowByrevID(ifstream& revFile,
                                        ifstream& offsetFile,
                                        int revId,
                                        const std::unordered_map<int, std::vector<int>>& revIdIndex) {
    std::vector<std::string> result;

    auto it = revIdIndex.find(revId);
    if (it == revIdIndex.end()) return result;

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
                  const unordered_map<int, vector<int>>& revIdIndex,
                  const string& outputFilename,
                  MyIndex& index)
{
    ifstream queryFile(queryFilename);
    ifstream csvFile(csvFilename, ios::binary);
    ifstream csvOffsetFile(csvOffsetFilename, ios::binary);
    ifstream revFile(revisionsFilename, ios::binary);
    ifstream revOffsetFile(revOffsetFilename, ios::binary);
    ofstream outFile(outputFilename);

    string line;
    int total = 0;

    while (getline(queryFile, line)) {
        stringstream ss(line);

        // Read query parameters
        float d;
        ss >> d >> d1 >> d2;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        float* query_data = new float[query_embedding.size()];
        copy(query_embedding.begin(), query_embedding.end(), query_data);

        // Perform KNN search
        auto indices = index.KNNWithDistanceUp(query_data, d);
        delete[] query_data;

        // Process each result row
        for (auto rowId : indices) {
            if(rowId==-1)continue;
            auto row = getRowByIndex(csvFile, csvOffsetFile, rowId);
            int oldId = stoi(extractColumns(row, {0})[0]);

            auto revRows = getRowByrevID(revFile, revOffsetFile, oldId, revIdIndex);

            for (const auto& revRow : revRows) {
                auto cols = extractColumns(revRow, {0, 4});
                int rev_id = stoi(cols[0]);
                const string& ts = cols[1];

                if (ts >= d1 && ts <= d2) {
                    outFile << rev_id << "\n";
                    total++;
                }
            }
        }
    }

    // Clean up
    queryFile.close();
    csvFile.close();
    csvOffsetFile.close();
    revFile.close();
    revOffsetFile.close();
    outFile.close();
}


void q(string query_size,MyIndex* index,unordered_map<int, std::vector<int>>& revIdIndex,int search_parameter) {
    
    string csvFilename = "../../database-generation/data_csv_files/text_csv_files/text.csv";
    string offsetFilename = "../../database-generation/offsets_files/text_offsets.bin";
    string revisionFilename = "../../database-generation/data_csv_files/revision_csv_files/revision_clean.csv";
    string revisionOffsetFilename = "../../database-generation/offsets_files/revision_offsets.bin";
    string revIdIndexFilename = "../../database-generation/index_files/rev_id_index.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    
    

    
    index->set_search_parameter(search_parameter);

    

    auto start = std::chrono::high_resolution_clock::now();
    
    fullPipeline(queryFilename,csvFilename,offsetFilename,revisionFilename,revisionOffsetFilename,revIdIndex,outputFilename,*index);

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
    queue<tuple<float, float*>>().swap(queryQueue);
    queue<int>().swap(rowIdQueue);
    queue<string>().swap(rowQueue);
    queue<int>().swap(oldIdQueue);
    queue<string>().swap(revisionRowQueue);
    queue<pair<int, string>>().swap(revTimestampQueue);
    queue<int>().swap(filteredRevIdQueue);

    

    

    
   

    
}
}
