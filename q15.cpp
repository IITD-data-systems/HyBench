#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <unordered_set>

using namespace std;

// Queues for each stage
queue<tuple<int, float*, float*>> queryQueue;
queue<vector<int>> filteredRowIdsQueue;
queue<vector<string>> rowsQueue;
queue<vector<int>> oldIdsQueue;

// Sync
mutex mtxQuery, mtxFiltered, mtxRows, mtxOldIds;
condition_variable cvQuery, cvFiltered, cvRows, cvOldIds;

bool no_more_queries = false;
int total_queries = 0, queries_processed = 0;

faiss::IndexHNSWFlat* index = nullptr;

// Simulated FAISS iKNN search
pair<vector<int>, vector<float>> KNN(faiss::IndexHNSWFlat& index, float* query, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

string getRowByIndex(const string& csvFile, const string& offsetFile, int rowId) {
    return "mocked_row_for_" + to_string(rowId);
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"1234", "some_old_text"};  // Dummy old_id and text
}

// Stage 1: Read k, embedding1 (cricket), embedding2 (batting)
void queryReaderThread(const string& queryFile) {
    ifstream file(queryFile);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k; ss >> k;
        vector<float> emb1, emb2;
        float val;
        for (int i = 0; i < 128 && ss >> val; ++i) emb1.push_back(val);
        for (int i = 0; i < 128 && ss >> val; ++i) emb2.push_back(val);

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

// Stage 2: iKNN + filtering
void iKNNFilterThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }

        auto [k, cricket, batting] = queryQueue.front(); queryQueue.pop();
        lock.unlock();

        auto [I1, D1] = KNN(*index, cricket, 2 * k);
        auto [I2, _]  = KNN(*index, batting, k);
        delete[] cricket; delete[] batting;

        unordered_set<int> battingSet(I2.begin(), I2.end());
        vector<pair<float, int>> filtered;
        for (int i = 0; i < 2 * k; ++i) {
            if (!battingSet.count(I1[i])) {
                filtered.emplace_back(D1[i], I1[i]);
            }
        }

        sort(filtered.begin(), filtered.end());
        vector<int> resultIds;
        for (auto& [dist, id] : filtered) {
            if ((int)resultIds.size() >= k) break;
            resultIds.push_back(id);
        }

        {
            lock_guard<mutex> lg(mtxFiltered);
            filteredRowIdsQueue.push(resultIds);
        }
        cvFiltered.notify_one();
    }
}

// Stage 3: Fetch rows by row_ids
void rowFetcherThread(const string& csvFile, const string& offsetFile) {
    while (true) {
        unique_lock<mutex> lock(mtxFiltered);
        cvFiltered.wait(lock, [] { return !filteredRowIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (filteredRowIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto ids = filteredRowIdsQueue.front(); filteredRowIdsQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int id : ids) {
            rows.push_back(getRowByIndex(csvFile, offsetFile, id));
        }

        {
            lock_guard<mutex> lg(mtxRows);
            rowsQueue.push(rows);
        }
        cvRows.notify_one();
    }
}

// Stage 4: Extract old_ids from rows
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

// Stage 5: Write output
void outputWriterThread(const string& outputFile) {
    ofstream out(outputFile);
    while (true) {
        unique_lock<mutex> lock(mtxOldIds);
        cvOldIds.wait(lock, [] { return !oldIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (oldIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto ids = oldIdsQueue.front(); oldIdsQueue.pop();
        queries_processed++;
        lock.unlock();

        for (int id : ids) {
            out << id << "\n";
        }
    }
    out.close();
}

// Main
int main() {
    string queryFile = "exclude_query.txt";
    string csvFile = "text.csv";
    string offsetFile = "text_offsets.bin";
    string outputFile = "exclude_output.txt";

    index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("text_index.faiss"));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNFilterThread);
    thread t3(rowFetcherThread, csvFile, offsetFile);
    thread t4(oldIdExtractorThread);
    thread t5(outputWriterThread, outputFile);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    return 0;
}
