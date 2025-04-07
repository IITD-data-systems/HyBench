#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

using namespace std;

// Shared Queues for Each Stage
queue<tuple<int, float, float, float*>> queryQueue;
queue<vector<int>> rowIdQueue;
queue<vector<string>> rowQueue;
queue<vector<vector<string>>> columnQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxColumn;
condition_variable cvQuery, cvRowId, cvRow, cvColumn;

// Control Variables
int num_queries_done = 0;
bool no_more_queries = false;
int total_queries = 0;

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
        int k;
        float d, d_star;
        ss >> k >> d >> d_star;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({k, d, d_star, query_data});
            total_queries++;
        }
        cvQuery.notify_one();
    }

    file.close();
    no_more_queries = true;
    cvQuery.notify_all();
}

// **New KNN Function: Returns Indices & Distances**


// **iKNN Thread**
void iKNNThread(faiss::IndexHNSWFlat &index) {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (queryQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        auto [k, d, d_star, query_data] = queryQueue.front();
        queryQueue.pop();
        lock.unlock();

        int optimalK = k;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(index, query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            if (it != distances.end()) {
                optimalK = (it - distances.begin()) + 1; // Set optimalK to first index where distance >= d
                break;
            }

            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(index, query_data, optimalK+k-1);
        
        // **Step 3: Filtering Indices Between d and d***

        vector<int> rowIds;
        for (size_t i = optimalK-1; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star and indices[i]!=-1) {
                rowIds.push_back(indices[i]);
            }
        }

        delete[] query_data;

        // **Pass rowIds to the next stage**
        lock_guard<mutex> lockRowId(mtxRowId);
        rowIdQueue.push(rowIds);
        cvRowId.notify_one();
    }
}

// **Row Extractor Thread**
void rowExtractorThread(const string& csvFilename, const string& offsetFilename) {
    while (true) {
        unique_lock<mutex> lock(mtxRowId);
        cvRowId.wait(lock, [] { return !rowIdQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (rowIdQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        vector<int> rowIds = rowIdQueue.front();
        rowIdQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int rowId : rowIds) {
            rows.push_back(getRowByIndex(csvFilename, offsetFilename, rowId));
        }

        lock_guard<mutex> lockRow(mtxRow);
        rowQueue.push(rows);
        cvRow.notify_one();
    }
}

// **Column Extractor Thread**
void columnExtractorThread(const vector<int>& columnIndices) {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !rowQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (rowQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        vector<string> rows = rowQueue.front();
        rowQueue.pop();
        lock.unlock();

        vector<vector<string>> extractedData;
        for (const string& row : rows) {
            extractedData.push_back(extractColumns(row, columnIndices));
        }

        lock_guard<mutex> lockColumn(mtxColumn);
        columnQueue.push(extractedData);
        cvColumn.notify_one();
    }
}

// **Output Writer Thread**
void outputThread(const string& outputFilename) {
    ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }

    while (true) {
        unique_lock<mutex> lock(mtxColumn);
        cvColumn.wait(lock, [] { return !columnQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (columnQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        vector<vector<string>> results = columnQueue.front();
        columnQueue.pop();
        lock.unlock();

        for (const auto& row : results) {
            outFile << "ID: " << row[0] << ", Text: " << row[1] << endl;
        }

        num_queries_done++;
    }

    outFile.close();
}

// **Main Function**
int main() {
    string queryFilename = "q4_queries.txt";
    string indexFilename = "old_text_embedding_index.faiss";
    string csvFilename = "text.csv";
    string offsetFilename = "text_offsets.bin";
    string outputFilename = "q4_output.txt";

    // **Load FAISS Index**
    faiss::IndexHNSWFlat* index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(indexFilename));

    vector<int> columnIndices = {0, 1};  // `old_id`, `old_text`

    thread t1(queryReaderThread, queryFilename);
    thread t2(iKNNThread, ref(*index));
    thread t3(rowExtractorThread, csvFilename, offsetFilename);
    thread t4(columnExtractorThread, columnIndices);
    thread t5(outputThread, outputFilename);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    delete index;
    return 0;
}
