#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

using namespace std;

// Shared Queues for Each Stage
queue<tuple<int, float, int64_t, int64_t, float*>> queryQueue;
queue<tuple<vector<int>, int64_t, int64_t>> rowIdQueue;
queue<tuple<vector<string>, int64_t, int64_t>> rowQueue;
queue<tuple<vector<int>, int64_t, int64_t>> oldIdQueue;
queue<tuple<vector<streampos>, int64_t, int64_t>> revisionOffsetQueue;
queue<tuple<vector<string>, int64_t, int64_t>> revisionRowQueue;
queue<tuple<vector<pair<int, int64_t>>, int64_t, int64_t>> revTimestampQueue;
queue<vector<int>> filteredRevIdQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxOldId, mtxRevOffset, mtxRevRow, mtxRevTime, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvOldId, cvRevOffset, cvRevRow, cvRevTime, cvOutput;

unordered_map<int, streampos> revIdToOffset;
bool no_more_queries = false;
int total_queries = 0;

void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k;
        float d;
        int64_t d1, d2;
        ss >> k >> d >> d1 >> d2;
        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);
        float* query_data = new float[query_embedding.size()];
        copy(query_embedding.begin(), query_embedding.end(), query_data);

        {
            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({k, d, d1, d2, query_data});
            total_queries++;
        }
        cvQuery.notify_one();
    }
    no_more_queries = true;
    cvQuery.notify_all();
}


void iKNNThread(faiss::IndexHNSWFlat& index) {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty()) break;

        auto [k, d, d1, d2, query_data] = queryQueue.front();
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
            if (finalDistances[i] <= d && indices[i] != -1) {
                rowIds.push_back(indices[i]);
            }
        }
        delete[] query_data;

        {
            lock_guard<mutex> lockRowId(mtxRowId);
            rowIdQueue.push({rowIds, d1, d2});
        }
        cvRowId.notify_one();
    }
}


void rowExtractorThread(const string& csvFilename, const string& offsetFilename) {
    while (true) {
        unique_lock<mutex> lock(mtxRowId);
        cvRowId.wait(lock, [] { return !rowIdQueue.empty(); });

        auto [rowIds, d1, d2] = rowIdQueue.front();
        rowIdQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int rowId : rowIds) {
            rows.push_back(getRowByIndex(csvFilename, offsetFilename, rowId));
        }

        {
            lock_guard<mutex> lockRow(mtxRow);
            rowQueue.push({rows, d1, d2});
        }
        {
            lock_guard<mutex> lockOldId(mtxOldId);
            oldIdQueue.push({rowIds, d1, d2});
        }
        cvRow.notify_one();
        cvOldId.notify_one();
    }
}

void oldIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !rowQueue.empty(); });

        auto [rows, d1, d2] = rowQueue.front();
        rowQueue.pop();
        lock.unlock();

        vector<int> oldIds;
        for (const string& row : rows) {
            oldIds.push_back(stoi(extractColumns(row, {0})[0]));
        }

        {
            lock_guard<mutex> lockOldId(mtxOldId);
            oldIdQueue.push({oldIds, d1, d2});
        }
        cvOldId.notify_one();
    }
}

void revisionOffsetLookupThread() {
    while (true) {
        unique_lock<mutex> lock(mtxOldId);
        cvOldId.wait(lock, [] { return !oldIdQueue.empty(); });

        auto [oldIds, d1, d2] = oldIdQueue.front();
        oldIdQueue.pop();
        lock.unlock();

        vector<streampos> offsets;
        for (int id : oldIds) offsets.push_back(revIdToOffset[id]);

        {
            lock_guard<mutex> lockRevOffset(mtxRevOffset);
            revisionOffsetQueue.push({offsets, d1, d2});
        }
        cvRevOffset.notify_one();
    }
}

void revisionRowExtractorThread(const string& revisionFilename) {
    while (true) {
        unique_lock<mutex> lock(mtxRevOffset);
        cvRevOffset.wait(lock, [] { return !revisionOffsetQueue.empty(); });

        auto [offsets, d1, d2] = revisionOffsetQueue.front();
        revisionOffsetQueue.pop();
        lock.unlock();

        ifstream file(revisionFilename);
        vector<string> rows;
        for (auto pos : offsets) {
            file.seekg(pos);
            string row;
            getline(file, row);
            rows.push_back(row);
        }

        {
            lock_guard<mutex> lockRevRow(mtxRevRow);
            revisionRowQueue.push({rows, d1, d2});
        }
        cvRevRow.notify_one();
    }
}

void revTimestampExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRevRow);
        cvRevRow.wait(lock, [] { return !revisionRowQueue.empty(); });

        auto [rows, d1, d2] = revisionRowQueue.front();
        revisionRowQueue.pop();
        lock.unlock();

        vector<pair<int, int64_t>> revData;
        for (const string& row : rows) {
            auto cols = extractColumns(row, {0, 1});
            revData.emplace_back(stoi(cols[0]), stoll(cols[1]));
        }

        {
            lock_guard<mutex> lockRevTime(mtxRevTime);
            revTimestampQueue.push({revData, d1, d2});
        }
        cvRevTime.notify_one();
    }
}

void filterThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRevTime);
        cvRevTime.wait(lock, [] { return !revTimestampQueue.empty(); });

        auto [revData, d1, d2] = revTimestampQueue.front();
        revTimestampQueue.pop();
        lock.unlock();

        vector<int> result;
        for (auto& [rev_id, ts] : revData) {
            if (ts >= d1 && ts <= d2) result.push_back(rev_id);
        }

        {
            lock_guard<mutex> lockOutput(mtxOutput);
            filteredRevIdQueue.push(result);
        }
        cvOutput.notify_one();
    }
}

void outputWriterThread(const string& outputFilename) {
    ofstream file(outputFilename);
    while (true) {
        unique_lock<mutex> lock(mtxOutput);
        cvOutput.wait(lock, [] { return !filteredRevIdQueue.empty(); });

        vector<int> revIds = filteredRevIdQueue.front();
        filteredRevIdQueue.pop();
        lock.unlock();

        for (int id : revIds) file << id << "\n";
    }
}

int main() {
    // Load revIdToOffset before starting threads
    // ...code to initialize revIdToOffset...

    faiss::IndexHNSWFlat* index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("index.faiss"));

    thread t1(queryReaderThread, "queries.txt");
    thread t2(iKNNThread, ref(*index));
    thread t3(rowExtractorThread, "text.csv", "text_offsets.bin");
    thread t4(oldIdExtractorThread);
    thread t5(revisionOffsetLookupThread);
    thread t6(revisionRowExtractorThread, "revisions.csv");
    thread t7(revTimestampExtractorThread);
    thread t8(filterThread);
    thread t9(outputWriterThread, "output.txt");

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join(); t9.join();
    delete index;
    return 0;
}
