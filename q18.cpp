#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <unordered_map>
using namespace std;

// Queues
queue<pair<float, float*>> queryQueue;
queue<vector<int>> filteredRowIdsQueue;
queue<vector<string>> textRowsQueue;
queue<vector<int>> oldIdsQueue;
queue<vector<string>> revisionRowsQueue;
queue<vector<pair<int, int>>> yearCountsQueue;

// Sync
mutex mtxQuery, mtxRowIds, mtxTextRows, mtxOldIds, mtxRevRows, mtxYearCount;
condition_variable cvQuery, cvRowIds, cvTextRows, cvOldIds, cvRevRows, cvYearCount;

bool no_more_queries = false;
int total_queries = 0, queries_processed = 0;

// FAISS index
faiss::IndexHNSWFlat* textIndex = nullptr;

// Dummy mappings
unordered_map<int, string> rowIdToOffsetText; // row_id -> offset string (text table)
unordered_map<int, string> revIdToOffset;     // rev_id -> offset string (revision table)

// File read placeholders
string getRowByIndex(const string& csv, const string& offset, int rowId) {
    return "row_for_" + to_string(rowId); // Dummy
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"12345"}; // Dummy value (e.g., old_id or timestamp)
}

pair<vector<int>, vector<float>> KNN(faiss::IndexHNSWFlat& index, float* query, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

// Stage 1: Read d and embedding
void queryReaderThread(const string& queryFile) {
    ifstream file(queryFile);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        float d;
        ss >> d;
        vector<float> emb;
        float val;
        while (ss >> val) emb.push_back(val);
        float* e = new float[128];
        copy(emb.begin(), emb.end(), e);

        {
            lock_guard<mutex> lg(mtxQuery);
            queryQueue.push({d, e});
            total_queries++;
        }
        cvQuery.notify_one();
    }
    no_more_queries = true;
    cvQuery.notify_all();
}

// Stage 2: iKNN with filter on distance < d
void iKNNFilterThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }

        auto [d, emb] = queryQueue.front(); queryQueue.pop();
        lock.unlock();

        int currK = 10;
        vector<int> filtered;
        while (true) {
            auto [I, D] = KNN(*textIndex, emb, currK);
            filtered.clear();
            for (int i = 0; i < currK; ++i) {
                if (D[i] < d) filtered.push_back(I[i]);
            }
            if (filtered.size() >= 10 || currK >= textIndex->ntotal) break;
            currK *= 2;
        }
        delete[] emb;

        {
            lock_guard<mutex> lg(mtxRowIds);
            filteredRowIdsQueue.push(filtered);
        }
        cvRowIds.notify_one();
    }
}

// Stage 3: Get rows from text.csv
void textRowFetcherThread(const string& csv, const string& offset) {
    while (true) {
        unique_lock<mutex> lock(mtxRowIds);
        cvRowIds.wait(lock, [] { return !filteredRowIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (filteredRowIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto ids = filteredRowIdsQueue.front(); filteredRowIdsQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int id : ids)
            rows.push_back(getRowByIndex(csv, offset, id));

        {
            lock_guard<mutex> lg(mtxTextRows);
            textRowsQueue.push(rows);
        }
        cvTextRows.notify_one();
    }
}

// Stage 4: Extract old_id
void oldIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxTextRows);
        cvTextRows.wait(lock, [] { return !textRowsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (textRowsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rows = textRowsQueue.front(); textRowsQueue.pop();
        lock.unlock();

        vector<int> revIds;
        for (auto& row : rows) {
            auto cols = extractColumns(row, {0});
            revIds.push_back(stoi(cols[0]));
        }

        {
            lock_guard<mutex> lg(mtxOldIds);
            oldIdsQueue.push(revIds);
        }
        cvOldIds.notify_one();
    }
}

// Stage 5: Get revision rows
void revisionRowFetcherThread(const string& revCsv, const string& offset) {
    while (true) {
        unique_lock<mutex> lock(mtxOldIds);
        cvOldIds.wait(lock, [] { return !oldIdsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (oldIdsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto revIds = oldIdsQueue.front(); oldIdsQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int rid : revIds)
            rows.push_back(getRowByIndex(revCsv, offset, rid));

        {
            lock_guard<mutex> lg(mtxRevRows);
            revisionRowsQueue.push(rows);
        }
        cvRevRows.notify_one();
    }
}

// Stage 6: Extract rev_id and rev_timestamp → Stage 7: Map[year]++
void timestampProcessorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRevRows);
        cvRevRows.wait(lock, [] { return !revisionRowsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (revisionRowsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto rows = revisionRowsQueue.front(); revisionRowsQueue.pop();
        lock.unlock();

        unordered_map<int, int> yearMap;
        for (auto& row : rows) {
            auto cols = extractColumns(row, {0, 1}); // rev_id, rev_timestamp
            string ts = cols[1];
            int year = stoi(ts.substr(0, 4));
            yearMap[year]++;
        }

        vector<pair<int, int>> result(yearMap.begin(), yearMap.end());

        {
            lock_guard<mutex> lg(mtxYearCount);
            yearCountsQueue.push(result);
        }
        cvYearCount.notify_one();
    }
}

// Stage 8: Write sorted output
void outputWriterThread(const string& outputFile) {
    ofstream out(outputFile);
    while (true) {
        unique_lock<mutex> lock(mtxYearCount);
        cvYearCount.wait(lock, [] { return !yearCountsQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (yearCountsQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }

        auto vec = yearCountsQueue.front(); yearCountsQueue.pop();
        queries_processed++;
        lock.unlock();

        sort(vec.begin(), vec.end());
        for (auto& [year, count] : vec)
            out << year << "," << count << "\n";
    }
    out.close();
}

int main() {
    string queryFile = "query.txt";
    string textCsv = "text.csv";
    string textOffset = "text_offsets.bin";
    string revCsv = "revision.csv";
    string revOffset = "revision_offsets.bin";
    string output = "year_counts.txt";

    textIndex = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("text_embedding_index.faiss"));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNFilterThread);
    thread t3(textRowFetcherThread, textCsv, textOffset);
    thread t4(oldIdExtractorThread);
    thread t5(revisionRowFetcherThread, revCsv, revOffset);
    thread t6(timestampProcessorThread);
    thread t7(outputWriterThread, output);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join();
    return 0;
}
