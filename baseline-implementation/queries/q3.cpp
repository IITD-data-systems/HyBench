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

namespace q3{
// Shared Queues for Each Stage
string query_number = "3";
queue<tuple<int,int, float*>> queryQueue;
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



// **Stage 1: Read Queries from File**
void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    if (!file.is_open()) {
        cerr << "Error: Could not open query file!" << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k,page_len;
        ss >> k >> page_len;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            
            queryQueue.push({k,page_len, query_data});
            
        }
        
    }
    
    file.close();
    no_more_queries1 = true;
    
    
    
} 
 
// **Stage 2: iKNN Search**
void iKNNThread(MyIndex &index) {
    string csvFilename = "../../data_csv_files/page_csv_files/page_extra.csv";
    string offsetFilename = "../../offsets_files/page_extra_offsets.bin";
    ifstream csvFile(csvFilename, std::ios::binary);
    ifstream offsetFile(offsetFilename, std::ios::binary);
    while (true) {
        

        if (queryQueue.empty() && no_more_queries1 ) break;

        auto [k,page_len, query_data] = queryQueue.front();
        queryQueue.pop();
        
        int k_temp=3*k;
        while(true){
            auto indices = index.KNNWithIndicesOnly(query_data, k_temp);
            int count=0;
            vector<int> final;
            for(auto x : indices){
                if(x==-1)continue;
                int len = get_page_len(x,csvFile,offsetFile);
                if (len<page_len){
                    count++;
                    final.push_back(x);
                }
                if (count==k){
                    for (int i = 0; i < k; i++) {
                        if (final[i]==-1)continue;
                        
                        rowIdQueue.push(final[i]);
                        
                        
                    }
                    break;
                }
            }
            if(count==k)break;
            k_temp*=2;
        }
        
        
        delete[] query_data;

        
        
        
    }
    no_more_queries2=true;
    
    csvFile.close();
    offsetFile.close();

}

// **Stage 3: Extract Rows from CSV**
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

// **Stage 4: Extract Specific Columns**
void columnExtractorThread(const vector<int>& columnIndices) {
    
    while (true) {
        

        if (rowQueue.empty() && no_more_queries3) break;

        string row = rowQueue.front();
        rowQueue.pop();
        

        vector<string> extractedData=extractColumns(row, columnIndices);
        
        
        columnQueue.push(extractedData);
        
    }
    no_more_queries4=true;
    
}

// **Stage 5: Write Output to File**
void outputThread(const string& outputFilename, const string&outputFilename2) {
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
        
        
        outFile2 << result[1] << "\n";
        outFile << result[0] << "\n";
        
        
    }

    outFile.close();
    outFile2.close();
    

}


void q(string query_size,MyIndex* index,int search_parameter) {
    
    string csvFilename = "../../database-generation/data_csv_files/page_csv_files/page.csv";
    string offsetFilename = "../../database-generation/offsets_files/page_offsets.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename2 = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_titles_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";

    
    
    
    
    // **Column Indices to Extract**
    vector<int> columnIndices = {0, 1};  // `page_id`, `page_title`


    // **Launch Threads**
    index->set_search_parameter(search_parameter);  
    

    auto start = std::chrono::high_resolution_clock::now();
    queryReaderThread(queryFilename);
    iKNNThread(*index);
    rowExtractorThread(csvFilename, offsetFilename);
    columnExtractorThread(columnIndices);
    outputThread(outputFilename, outputFilename2);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Total execution time for query " + query_number + " subtype" + query_size + ": " << duration.count()*1000 << " seconds\n";
    


    ofstream time_file("../../output-files/final_queries_time",ios::app);
    time_file << index->index_kind() << " " << index->metric_type() << " " <<  query_number  << " "<< query_size << " " << duration.count() <<endl;
    time_file.close();

    no_more_queries1 = false;
    no_more_queries2 = false;
    no_more_queries3 = false;
    no_more_queries4 = false;
    
    
}
}
