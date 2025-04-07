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
queue<tuple<int, float, float, float*>> queryQueue;
queue<vector<int>> finalRowIdsQueue;
queue<vector<string>> rowsQueue;
queue<vector<string>> pageIdRowsQueue;
queue<vector<tuple<string, string, int>>> finalOutputQueue;

// Sync
mutex mtxQuery, mtxFinal, mtxRows, mtxPageId, mtxOutput;
condition_variable cvQuery, cvFinal, cvRows, cvPageId, cvOutput;

bool no_more_queries = false;
int total_queries = 0, queries_processed = 0;

// FAISS index
faiss::IndexHNSWFlat* pageTitleIndex = nullptr;

// Dummy maps
unordered_map<int, vector<int>> pageToRevMap;     // page_id -> rev_ids
unordered_map<int, string> revIdToOffset;         // rev_id -> offset line
unordered_map<int, string> pageIdToTitle;         // page_id -> page_title
unordered_map<int, string> pageIdToRedirect;      // page_id -> is_redirect

// Dummy file access
string getRowByIndex(const string& csv, const string& offsetFile, int rowId) {
    return "row_for_" + to_string(rowId); // Replace with real file logic
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"1234", "some_title", "0"}; // Replace with real parsing
}

string getRevTimestamp(int revId) {
    return "20230407120000"; // Placeholder timestamp
}

// Helper: FAISS iKNN
pair<vector<int>, vector<float>> KNN(faiss::IndexHNSWFlat& index, float* query, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

// Stage 1: Read k, d, d*, embedding
void queryReaderThread(const string& queryFile) {
    ifstream file(queryFile);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k; float d, d_star;
        ss >> k >> d >> d_star;
        vector<float> emb;
        float val;
        for (int i = 0; i < 128 && ss >> val; ++i) emb.push_back(val);
        float* e = new float[128];
        copy(emb.begin(), emb.end(), e);

        {
            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({k, d, d_star, e});
            total_queries++;
        }
        cvQuery.notify_one();
    }
    no_more_queries = true;
    cvQuery.notify_all();
}

// Stage 2: Find optimalK and return filtered top-K results by distance ∈ [d, d*]
void iKNNFilterThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }

        auto [k, d, d_star, emb] = queryQueue.front(); queryQueue.pop();
        lock.unlock();

        int currK = k;
        vector<int> finalIds;

        while (true) {
            auto [I, D] = KNN(*pageTitleIndex, emb, currK);
            finalIds.clear();
            for (int i = 0; i < currK; ++i) {
                if (D[i] >= d && D[i] <= d_star)
                    finalIds.push_back(I[i]);
            }
            if ((int)finalIds.size() >= k || currK >= pageTitleIndex->ntotal) break;
            currK *= 2;
        }

        finalIds.resize(min((int)finalIds.size(), k));
        delete[] emb;

        {
            lock_guard<mutex> lg(mtxFinal);
            finalRowIdsQueue.push(finalIds);
        }
        cvFinal.notify_one();
    }
}

// Stage 3: Get rows from page table
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

// Stage 4: Extract page_id
void pageIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRows);
        cvRows.wait(lock, [] { return !rowsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (rowsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rows = rowsQueue.front(); rowsQueue.pop();
        lock.unlock();

        vector<string> pageInfos;
        for (const auto& row : rows) {
            auto cols = extractColumns(row, {0}); // Assume page_id is at index 0
            pageInfos.push_back(cols[0]); // push back page_id
        }

        {
            lock_guard<mutex> lg(mtxPageId);
            pageIdRowsQueue.push(pageInfos);
        }
        cvPageId.notify_one();
    }
}

// Stage 5: Lookup rev_id → timestamps → final aggregation
void revisionAggregatorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxPageId);
        cvPageId.wait(lock, [] { return !pageIdRowsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (pageIdRowsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto pageIds = pageIdRowsQueue.front(); pageIdRowsQueue.pop();
        lock.unlock();

        vector<tuple<string, string, int>> result;
        for (const string& pidStr : pageIds) {
            int pid = stoi(pidStr);
            const auto& revs = pageToRevMap[pid];

            string minTs = "99999999999999", maxTs = "00000000000000";
            for (int rid : revs) {
                string ts = getRevTimestamp(rid);
                minTs = min(minTs, ts);
                maxTs = max(maxTs, ts);
            }

            result.emplace_back(minTs, maxTs, (int)revs.size());
        }

        {
            lock_guard<mutex> lg(mtxOutput);
            finalOutputQueue.push(result);
        }
        cvOutput.notify_one();
    }
}

// Stage 6: Write final output
void outputWriterThread(const string& outputFile) {
    ofstream out(outputFile);
    while (true) {
        unique_lock<mutex> lock(mtxOutput);
        cvOutput.wait(lock, [] { return !finalOutputQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (finalOutputQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rows = finalOutputQueue.front(); finalOutputQueue.pop();
        queries_processed++;
        lock.unlock();

        for (const auto& [created, modified, count] : rows) {
            // Example output: title, created, modified, rev_count, is_redirect
            out << "title_placeholder" << "," << created << "," << modified << "," << count << "," << "0" << "\n";
        }
    }
    out.close();
}

int main() {
    string queryFile = "embedding_query.txt";
    string csvFile = "page.csv";
    string offsetFile = "page_offsets.bin";
    string outputFile = "knn_filtered_output.txt";

    pageTitleIndex = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("page_title_index.faiss"));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNFilterThread);
    thread t3(rowFetcherThread, csvFile, offsetFile);
    thread t4(pageIdExtractorThread);
    thread t5(revisionAggregatorThread);
    thread t6(outputWriterThread, outputFile);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join();
    return 0;
}
