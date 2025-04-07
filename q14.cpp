#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <set>

using namespace std;

// Shared Queues
queue<tuple<int, float*, float*>> queryQueue;
queue<vector<pair<float, int>>> unionQueue;
queue<set<int>> topKRowIdsQueue;
queue<vector<string>> rowsQueue;
queue<vector<int>> oldIdsQueue;

// Mutexes and Condition Variables
mutex mtxQuery, mtxUnion, mtxTopK, mtxRows, mtxOldIds;
condition_variable cvQuery, cvUnion, cvTopK, cvRows, cvOldIds;

bool no_more_queries = false;
int total_queries = 0;
int queries_processed = 0;

// FAISS Index
faiss::IndexHNSWFlat* index = nullptr;

// Mock utils
pair<vector<int>, vector<float>> KNN(faiss::IndexHNSWFlat& index, float* query_data, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query_data, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

string getRowByIndex(const string& csvFilename, const string& offsetFilename, int rowId) {
    return "mocked_row";
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"1234", "mocked_old_text"}; // 0: old_id, 1: old_text
}

// Stage 1: Read k, embedding1, embedding2
void queryReaderThread(const string& queryFile) {
    ifstream file(queryFile);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k; ss >> k;
        vector<float> emb1, emb2;
        float val;
        for (int i = 0; i < 128; ++i && ss >> val) emb1.push_back(val);
        for (int i = 0; i < 128; ++i && ss >> val) emb2.push_back(val);

        float* e1 = new float[128], *e2 = new float[128];
        copy(emb1.begin(), emb1.end(), e1);
        copy(emb2.begin(), emb2.end(), e2);

        {
            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({k, e1, e2});
            total_queries++;
        }
        cvQuery.notify_one();
    }
    no_more_queries = true;
    cvQuery.notify_all();
}

// Stage 2: Do iKNN for both embeddings, push union (2k results)
void iKNNThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }

        auto [k, e1, e2] = queryQueue.front(); queryQueue.pop();
        lock.unlock();

        auto [I1, D1] = KNN(*index, e1, k);
        auto [I2, D2] = KNN(*index, e2, k);

        delete[] e1; delete[] e2;

        vector<pair<float, int>> results;
        for (int i = 0; i < k; ++i) results.emplace_back(D1[i], I1[i]);
        for (int i = 0; i < k; ++i) results.emplace_back(D2[i], I2[i]);

        {
            lock_guard<mutex> lg(mtxUnion);
            unionQueue.push(results);
        }
        cvUnion.notify_one();
    }
}

// Stage 3: Take union, sort by distance, get top k unique rowIds
void topKSelectorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxUnion);
        cvUnion.wait(lock, [] { return !unionQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (unionQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto results = unionQueue.front(); unionQueue.pop();
        lock.unlock();

        sort(results.begin(), results.end());
        set<int> selected;
        for (auto& [dist, id] : results) {
            if (selected.size() >= (size_t)results.size() / 2) break;
            selected.insert(id);
        }

        {
            lock_guard<mutex> lg(mtxTopK);
            topKRowIdsQueue.push(selected);
        }
        cvTopK.notify_one();
    }
}

// Stage 4: Get rows from rowIds
void rowFetcherThread(const string& csvFile, const string& offsetFile) {
    while (true) {
        unique_lock<mutex> lock(mtxTopK);
        cvTopK.wait(lock, [] { return !topKRowIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (topKRowIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rowIds = topKRowIdsQueue.front(); topKRowIdsQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int id : rowIds) {
            rows.push_back(getRowByIndex(csvFile, offsetFile, id));
        }

        {
            lock_guard<mutex> lg(mtxRows);
            rowsQueue.push(rows);
        }
        cvRows.notify_one();
    }
}

// Stage 5: Extract old_id from rows
void oldIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRows);
        cvRows.wait(lock, [] { return !rowsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (rowsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rows = rowsQueue.front(); rowsQueue.pop();
        lock.unlock();

        vector<int> oldIds;
        for (const auto& row : rows) {
            oldIds.push_back(stoi(extractColumns(row, {0})[0]));
        }

        {
            lock_guard<mutex> lg(mtxOldIds);
            oldIdsQueue.push(oldIds);
        }
        cvOldIds.notify_one();
    }
}

// Stage 6: Write old_id to file
void outputWriterThread(const string& outputFile) {
    ofstream out(outputFile);
    while (true) {
        unique_lock<mutex> lock(mtxOldIds);
        cvOldIds.wait(lock, [] { return !oldIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (oldIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto oldIds = oldIdsQueue.front(); oldIdsQueue.pop();
        queries_processed++;
        lock.unlock();

        sort(oldIds.begin(), oldIds.end());
        for (int id : oldIds) {
            out << id << "\n";
        }
    }
    out.close();
}

// Main
int main() {
    string queryFile = "union_query.txt";
    string csvFile = "text.csv";
    string offsetFile = "text_offsets.bin";
    string outputFile = "union_output.txt";

    index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("text_index.faiss"));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNThread);
    thread t3(topKSelectorThread);
    thread t4(rowFetcherThread, csvFile, offsetFile);
    thread t5(oldIdExtractorThread);
    thread t6(outputWriterThread, outputFile);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join();
    return 0;
}
