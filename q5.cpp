#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

using namespace std;

// Shared Queues for Each Stage
queue<tuple<int, int, float*>> queryQueue;
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
        int l, r;
        ss >> l >> r;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({l, r, query_data});
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
// **iKNN Thread for Rank Range Query**
void iKNNThread(faiss::IndexHNSWFlat &index) {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (queryQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        auto [l, r, query_data] = queryQueue.front();
        queryQueue.pop();
        lock.unlock();

        // **Step 1: Perform KNN for k = r**
        vector<int> indices = KNNWithIndicesOnly(index, query_data, r);
        
        // **Step 2: Extract Range [l-1, r]**
        vector<int> rowIds(indices.begin() + (l - 1), indices.begin() + r);

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
    string queryFilename = "q5_queries.txt";
    string indexFilename = "page_title_embedding_index.faiss";
    string csvFilename = "page.csv";
    string offsetFilename = "page_offsets.bin";
    string outputFilename = "q5_output.txt";

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
