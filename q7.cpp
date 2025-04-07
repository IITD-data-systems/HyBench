#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

using namespace std;

// Shared Queues for Each Stage
queue<tuple<int, float, float, float*>> queryQueue;
queue<vector<int>> rowIdQueue;
queue<vector<string>> rowQueue;
queue<vector<int>> pageIdQueue;
queue<vector<int>> actorIdQueue;
queue<vector<pair<int, int>>> actorCountQueue;
queue<vector<pair<int, int>>> outputQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxPageId, mtxActorId, mtxActorCount, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvPageId, cvActorId, cvActorCount, cvOutput;

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

        while (true) {
            distances = KNNWithDistancesOnly(index, query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            if (it != distances.end()) {
                optimalK = (it - distances.begin()) + 1;
                break;
            }
            optimalK *= 2;
        }

        auto [indices, finalDistances] = KNNWithIndicesAndDistances(index, query_data, optimalK + k - 1);

        vector<int> rowIds;
        for (size_t i = optimalK - 1; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i] != -1) {
                rowIds.push_back(indices[i]);
            }
        }

        delete[] query_data;

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

// **Column Extractor Thread (Extract Page IDs)**
void columnExtractorThread(const vector<int>& columnIndices) {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !rowQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (rowQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        vector<string> rows = rowQueue.front();
        rowQueue.pop();
        lock.unlock();

        vector<int> pageIds;
        for (const string& row : rows) {
            pageIds.push_back(stoi(extractColumns(row, columnIndices)[0]));
        }

        lock_guard<mutex> lockPageId(mtxPageId);
        pageIdQueue.push(pageIds);
        cvPageId.notify_one();
    }
}

// **Row Extractor Thread for Revisions**
void rowExtractorThread1(const string& revisionsFilename, const string& offsetFilename) {
    while (true) {
        unique_lock<mutex> lock(mtxPageId);
        cvPageId.wait(lock, [] { return !pageIdQueue.empty(); });

        vector<int> pageIds = pageIdQueue.front();
        pageIdQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int pageId : pageIds) {
            rows.push_back(getRowByPageID(revisionsFilename, offsetFilename, pageId));
        }

        lock_guard<mutex> lockRow(mtxActorId);
        actorIdQueue.push(rows);
        cvActorId.notify_one();
    }
}

// **Column Extractor Thread (Extract Actor IDs)**
void columnExtractorThread1() {
    while (true) {
        unique_lock<mutex> lock(mtxActorId);
        cvActorId.wait(lock, [] { return !actorIdQueue.empty(); });

        vector<string> rows = actorIdQueue.front();
        actorIdQueue.pop();
        lock.unlock();

        vector<int> actorIds;
        for (const string& row : rows) {
            actorIds.push_back(stoi(extractColumns(row, {1})[0]));
        }

        lock_guard<mutex> lockActorCount(mtxActorCount);
        actorCountQueue.push(actorIds);
        cvActorCount.notify_one();
    }
}

// **Actor Aggregator Thread**
void actorAggregatorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxActorCount);
        cvActorCount.wait(lock, [] { return !actorCountQueue.empty(); });

        vector<int> actorIds = actorCountQueue.front();
        actorCountQueue.pop();
        lock.unlock();

        unordered_map<int, int> actorCounts;
        for (int actorId : actorIds) {
            actorCounts[actorId]++;
        }

        vector<pair<int, int>> sortedActors(actorCounts.begin(), actorCounts.end());
        sort(sortedActors.begin(), sortedActors.end(), [](auto& a, auto& b) { return a.second > b.second; });

        lock_guard<mutex> lockOutput(mtxColumn);
        columnQueue.push(sortedActors);
        cvColumn.notify_one();
    }
}

void outputWriterThread(const string& outputFilename) {
    ofstream file(outputFilename);
    if (!file.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }
    
    while (true) {
        unique_lock<mutex> lock(mtxOutput);
        cvOutput.wait(lock, [] { return !outputQueue.empty() || (no_more_queries && num_queries_done == total_queries); });
        
        if (outputQueue.empty() && no_more_queries && num_queries_done == total_queries) break;
        
        vector<pair<int, int>> sortedActors = outputQueue.front();
        outputQueue.pop();
        lock.unlock();
        
        for (const auto& [actorId, count] : sortedActors) {
            file << actorId << "," << count << "\n";
        }
    }
    
    file.close();
}


int main() {
    string queryFilename = "q7_queries.txt";
    string indexFilename = "old_text_embedding_index.faiss";
    string csvFilename = "text.csv";
    string offsetFilename = "text_offsets.bin";
    string revisionFilename = "revisions.csv";
    string revisionOffsetFilename = "revision_offsets.bin";
    string outputFilename = "q6_output.txt";

    // **Load FAISS Index**
    faiss::IndexHNSWFlat* index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(indexFilename));

    vector<int> columnIndices = {0, 1};  // `old_id`, `old_text`
    vector<int> revisionColumnIndices = {1, 2}; // `rev_page`, `rev_actor`

    thread t1(queryReaderThread, queryFilename);
    thread t2(iKNNThread, ref(*index));
    thread t3(rowExtractorThread, csvFilename, offsetFilename);
    thread t4(columnExtractorThread, columnIndices);
    thread t5(rowExtractorThread1, revisionFilename, revisionOffsetFilename);
    thread t6(columnExtractorThread1, revisionColumnIndices);
    thread t7(actorAggregatorThread);
    thread t8(outputThread, outputFilename);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();

    delete index;
    return 0;
}
