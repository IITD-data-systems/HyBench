#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

using namespace std;

// Shared Queues
queue<pair<float, float*>> queryQueue;
queue<unordered_map<int, vector<pair<float, int>>>> yearRowIdQueue;
queue<unordered_map<int, vector<string>>> yearRowQueue;
queue<unordered_map<int, vector<int>>> yearOldIdQueue;

// Mutexes and Condition Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxOldId, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvOldId, cvOutput;

bool no_more_queries = false;
int total_queries = 0;
int queries_processed = 0;

unordered_map<int, faiss::IndexHNSWFlat*> yearIndices; // year -> faiss index
unordered_map<int, string> yearCsvFiles;               // year -> csv file
unordered_map<int, string> yearOffsetFiles;            // year -> offset file
unordered_map<int, unordered_map<int, int>> yearOffsets; // year -> (rowId -> offset)

// Mock utilities
pair<vector<int>, vector<float>> KNNWithIndicesAndDistances(faiss::IndexHNSWFlat& index, float* query_data, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query_data, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

string getRowByIndex(const string& csvFilename, const string& offsetFilename, int rowId) {
    return "mocked_row"; // Replace with actual file logic
}

vector<string> extractColumns(const string& row, const vector<int>& indices) {
    return {"mocked_old_id"}; // Replace with actual parsing logic
}

// Stage 1: Query Reader — updated to read `k` instead of `d`
void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k; ss >> k;
        vector<float> q;
        float val; while (ss >> val) q.push_back(val);
        if (!q.empty()) {
            float* data = new float[q.size()];
            copy(q.begin(), q.end(), data);
            {
                lock_guard<mutex> lock(mtxQuery);
                queryQueue.push({(float)k, data});
                total_queries++;
            }
            cvQuery.notify_one();
        }
    }
    file.close();
    no_more_queries = true;
    cvQuery.notify_all();
}

// Stage 2: iKNN on each year index
void iKNNThread() {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) {
            if (no_more_queries) break;
            continue;
        }
        auto [k_f, q] = queryQueue.front(); queryQueue.pop();
        int k = (int)k_f;
        lock.unlock();

        unordered_map<int, vector<pair<float, int>>> yearToResults;

        for (auto& [year, index] : yearIndices) {
            auto [I, D] = KNNWithIndicesAndDistances(*index, q, 100); // Retrieve max possible
            vector<pair<float, int>> filtered;
            for (int i = 0; i < D.size(); ++i) {
                filtered.emplace_back(D[i], I[i]);
            }
            sort(filtered.begin(), filtered.end());
            if (!filtered.empty()) {
                yearToResults[year] = vector<pair<float, int>>(filtered.begin(), filtered.begin() + min((int)filtered.size(), k));
            }
        }

        delete[] q;

        {
            lock_guard<mutex> lg(mtxRowId);
            yearRowIdQueue.push(yearToResults);
        }
        cvRowId.notify_one();
    }
}

// Stage 3: Get rows for each rowId
void rowExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRowId);
        cvRowId.wait(lock, [] { return !yearRowIdQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (yearRowIdQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }
        auto yearMap = yearRowIdQueue.front(); yearRowIdQueue.pop();
        lock.unlock();

        unordered_map<int, vector<string>> yearRows;
        for (auto& [year, results] : yearMap) {
            for (auto& [dist, rowId] : results) {
                yearRows[year].push_back(getRowByIndex(yearCsvFiles[year], yearOffsetFiles[year], rowId));
            }
        }

        {
            lock_guard<mutex> lg(mtxRow);
            yearRowQueue.push(yearRows);
        }
        cvRow.notify_one();
    }
}

// Stage 4: Extract old_id for each row
void oldIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !yearRowQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (yearRowQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }
        auto rowsByYear = yearRowQueue.front(); yearRowQueue.pop();
        lock.unlock();

        unordered_map<int, vector<int>> yearOldIds;
        for (auto& [year, rows] : rowsByYear) {
            for (auto& row : rows) {
                yearOldIds[year].push_back(stoi(extractColumns(row, {0})[0]));
            }
        }

        {
            lock_guard<mutex> lg(mtxOldId);
            yearOldIdQueue.push(yearOldIds);
        }
        cvOldId.notify_one();
    }
}

// Stage 5: Output Writer
void outputWriterThread(const string& outputFilename) {
    ofstream out(outputFilename);
    while (true) {
        unique_lock<mutex> lock(mtxOldId);
        cvOldId.wait(lock, [] { return !yearOldIdQueue.empty() || (no_more_queries && queries_processed == total_queries); });
        if (yearOldIdQueue.empty()) {
            if (no_more_queries && queries_processed == total_queries) break;
            continue;
        }
        auto results = yearOldIdQueue.front(); yearOldIdQueue.pop();
        queries_processed++;
        lock.unlock();

        vector<int> years;
        for (auto& [year, _] : results) years.push_back(year);
        sort(years.rbegin(), years.rend());

        for (int year : years) {
            vector<int>& ids = results[year];
            sort(ids.begin(), ids.end());
            for (int id : ids) out << year << "," << id << "\n";
        }
    }
    out.close();
}

int main() {
    string queryFile = "q11_queries.txt", outFile = "q11_output.txt";

    for (int y = 2020; y <= 2025; ++y) {
        string idxFile = "index_" + to_string(y) + ".faiss";
        yearIndices[y] = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(idxFile.c_str()));
        yearCsvFiles[y] = "text_" + to_string(y) + ".csv";
        yearOffsetFiles[y] = "text_offsets_" + to_string(y) + ".bin";
    }

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNThread);
    thread t3(rowExtractorThread);
    thread t4(oldIdExtractorThread);
    thread t5(outputWriterThread, outFile);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    return 0;
}
