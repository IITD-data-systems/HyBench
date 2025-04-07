#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// Queues
queue<tuple<int, float*, float*>> queryQueue;
queue<vector<int>> finalRowIdsQueue;
queue<vector<string>> rowsQueue;
queue<vector<int>> oldIdsQueue;

// Sync
mutex mtxQuery, mtxRows, mtxFinal, mtxOldIds;
condition_variable cvQuery, cvFinal, cvRows, cvOldIds;

bool no_more_queries = false;
int total_queries = 0, queries_processed = 0;

faiss::IndexHNSWFlat* index = nullptr;

// Helper: FAISS iKNN
pair<vector<int>, vector<float>> KNN(faiss::IndexHNSWFlat& index, float* query, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

// Dummy CSV row retriever
string getRowByIndex(const string& csvFile, const string& offsetFile, int rowId) {
    return "row_for_" + to_string(rowId); // Replace with actual file IO logic
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"1234", "some_old_text"};  // Dummy old_id and text
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

// Stage 2: Iterative iKNN + GREATEST distance filter
void iKNNIntersectionThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }

        auto [k, emb1, emb2] = queryQueue.front(); queryQueue.pop();
        lock.unlock();

        int currK = k;
        unordered_map<int, float> dist1, dist2;
        unordered_set<int> inter;

        while (true) {
            auto [I1, D1] = KNN(*index, emb1, currK);
            auto [I2, D2] = KNN(*index, emb2, currK);

            dist1.clear(); dist2.clear();
            for (int i = 0; i < currK; ++i) dist1[I1[i]] = D1[i];
            for (int i = 0; i < currK; ++i) dist2[I2[i]] = D2[i];

            inter.clear();
            for (const auto& p : dist1) {
                if (dist2.count(p.first)) inter.insert(p.first);
            }

            if ((int)inter.size() >= k) break;
            currK *= 2;
            if (currK > index->ntotal) break; // Avoid overflow
        }

        // Sort intersection by GREATEST(dist1, dist2)
        vector<tuple<float, int>> candidates;
        for (int id : inter) {
            float g = max(dist1[id], dist2[id]);
            candidates.emplace_back(g, id);
        }

        sort(candidates.begin(), candidates.end());

        vector<int> topK;
        for (int i = 0; i < min(k, (int)candidates.size()); ++i)
            topK.push_back(get<1>(candidates[i]));

        delete[] emb1;
        delete[] emb2;

        {
            lock_guard<mutex> lg(mtxFinal);
            finalRowIdsQueue.push(topK);
        }
        cvFinal.notify_one();
    }
}

// Stage 3: Get rows
void rowFetcherThread(const string& csvFile, const string& offsetFile) {
    while (true) {
        unique_lock<mutex> lock(mtxFinal);
        cvFinal.wait(lock, [] { return !finalRowIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (finalRowIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto ids = finalRowIdsQueue.front(); finalRowIdsQueue.pop();
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

// Stage 4: Extract old_id
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
    string queryFile = "intersection_query.txt";
    string csvFile = "text.csv";
    string offsetFile = "text_offsets.bin";
    string outputFile = "intersection_output.txt";

    index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("text_index.faiss"));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNIntersectionThread);
    thread t3(rowFetcherThread, csvFile, offsetFile);
    thread t4(oldIdExtractorThread);
    thread t5(outputWriterThread, outputFile);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    return 0;
}
