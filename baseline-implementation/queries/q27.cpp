#include <bits/stdc++.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>   // Needed for quantizer
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "../pipeline_stages.cpp"
#include <chrono>


using namespace std;

namespace q27{
string query_number="27";
// Shared Queues for Each Stage
queue<tuple<vector<pair<float, float>>, float*>> queryQueue;
queue<int> rowIdQueue;
queue<string> rowQueue;
queue<vector<string>> columnQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxColumn;
condition_variable cvQuery, cvRowId, cvRow, cvColumn;

// Control Variables

bool no_more_queries1 = false;
bool no_more_queries2 = false;
bool no_more_queries3 = false;
bool no_more_queries4 = false;



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
        
        
        vector<pair<float,float>> distance_ranges;
        int num_constraints;
        ss >> num_constraints;
        for(int i=0;i<num_constraints;i++){
            float d,d_star;
            ss >> d >> d_star;
            distance_ranges.push_back({d,d_star});
        }
        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            
            queryQueue.push({ distance_ranges, query_data});
            
        }
        
    }

    file.close();
    no_more_queries1 = true;
    
}

// **New KNN Function: Returns Indices & Distances**


// **iKNN Thread**
void iKNNThread(MyIndex &index) {
    while (true) {
        

        if (queryQueue.empty() && no_more_queries1) break;

        auto [distance_ranges, query_data] = queryQueue.front();
        queryQueue.pop();
        
        
        // **Step 3: Filtering Indices Between d and d***
        float ma=-1000000;
        float mi=1000000;
        for(auto& [x,y] : distance_ranges){
            mi=min(mi,x);
            ma=max(ma,y);
        }
        auto [indices,distances] = index.KNNWithDistanceUpDownWithDistances(query_data,mi,ma);


        for (int i = 0; i < indices.size(); i++) {
            
                if(indices[i]==-1)continue;
                if(!range_lie_identifier(distances[i],distance_ranges))continue;
                
                rowIdQueue.push(indices[i]);
                
        }

        delete[] query_data;

        // **Pass rowIds to the next stage**
        
    }
    no_more_queries2=true;
    
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

// **Column Extractor Thread**
void columnExtractorThread(const vector<int>& columnIndices) {
    while (true) {
        

        if (rowQueue.empty() && no_more_queries3) break;

        string row = rowQueue.front();
        rowQueue.pop();
        
        columnQueue.push(extractColumns(row, columnIndices));
        
    }
    no_more_queries4=true;
    
}
 
// **Output Writer Thread**
void outputThread(const string& outputFilename,  const string& outputFilename2) {
    ofstream outFile(outputFilename);
    ofstream outFile2(outputFilename2);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }

    while (true) {
        

        if (columnQueue.empty() && no_more_queries4) break;

        vector<string> result = columnQueue.front();
        columnQueue.pop();
        

        
        outFile << result[0]<< "\n";
        outFile2 << result[1] << "\n";
        
    }

    outFile.close();
    outFile2.close();
}

// **Main Function**
void q(string query_size,MyIndex* index,int search_parameter) {

    
    string csvFilename = "../../database-generation/data_csv_files/text_csv_files/text.csv";
    string offsetFilename = "../../database-generation/offsets_files/text_offsets.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename2 = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_titles_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    
    index->set_search_parameter(search_parameter);
    vector<int> columnIndices = {0, 1}; 
    

    auto start = std::chrono::high_resolution_clock::now();
    

    queryReaderThread(queryFilename);
    iKNNThread(*index);
    rowExtractorThread(csvFilename, offsetFilename);
    columnExtractorThread(columnIndices);
    outputThread(outputFilename, outputFilename2);

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
    
}
}
