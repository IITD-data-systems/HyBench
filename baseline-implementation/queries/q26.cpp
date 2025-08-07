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

namespace q26{
string query_number = "26";
// Shared Queues for Each Stage
queue<pair<vector<int>, float*>> queryQueue;
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


int dim=384;

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
        int num_ranks;
        ss >> num_ranks;
        vector<int> rank_list;
        for(int i=0;i<num_ranks;i++){
            int rank;
            ss>>rank;
            rank_list.push_back(rank);
        }

        
        float* query_data=new float[dim];
        float value;
        for(int i=0;i<dim;i++){
            ss >> value;query_data[i]=value;
        }
        if (true) {
            
            queryQueue.push({rank_list, query_data});
        }
        
    }

    file.close();
    no_more_queries1 = true;
    
}

// **Stage 2: iKNN Search**
void iKNNThread(MyIndex &index) {
    while (true) {
        

        if (queryQueue.empty() && no_more_queries1) break;

        auto [rank_list, query_data] = queryQueue.front();
        queryQueue.pop();
        
        int rank_max=rank_list[rank_list.size()-1];
        auto indices = index.KNNWithIndicesOnly(query_data, rank_max);	
        delete[] query_data;

        
        for (auto x : rank_list) {
            if(indices[x-1]==-1)continue;
            
            rowIdQueue.push(indices[x-1]);
            
        }
        
    }
    no_more_queries2=true;
    
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
void outputThread(const string& outputFilename, const string& outputFilename2) {
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


void q(string query_size,MyIndex* index,int search_parameter) {
    
    string csvFilename = "../../database-generation/data_csv_files/text_csv_files/text.csv";
    string offsetFilename = "../../database-generation/offsets_files/text_offsets.bin";
    const string queryFilename = "../../query-generation/q" + query_number + "_queries/q" + query_number + "_queries_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    string outputFilename2 = "../../output-files/baseline_queries_output/q" + query_number + "/q" + query_number + "_output_titles_" + index->index_kind() + "_" + index->metric_type() + "_" + query_size + ".txt";
    // **Load FAISS Index**
    ifstream infile("../../database-generation/dim");
	infile >> dim;
    
    index->set_search_parameter(search_parameter);
    // **Column Indices to Extract**
    
    vector<int> columnIndices = {0, 1};  
    
    // **Launch Threads**
    auto start = std::chrono::high_resolution_clock::now();
    

    queryReaderThread(queryFilename);
    iKNNThread(*index);
    rowExtractorThread(csvFilename, offsetFilename);
    columnExtractorThread(columnIndices);
    outputThread(outputFilename, outputFilename2);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time for query " + query_number + " subtype " + query_size + ": " << duration.count()*1000 << " seconds\n";

    ofstream time_file("../../output-files/final_queries_time",ios::app);
    time_file << index->index_kind() << " " << index->metric_type() << " " << query_number << " "<< query_size << " " << duration.count() <<endl;
    time_file.close();
    
    no_more_queries1 = false;
    no_more_queries2 = false;
    no_more_queries3 = false;
    no_more_queries4 = false;
    
}
}
