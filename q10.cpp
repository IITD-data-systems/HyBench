#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

using namespace std;

// Shared Queues
queue<tuple<int, float*>> queryQueue;
queue<vector<int>> rowIdQueue;
queue<vector<string>> rowQueue;
queue<vector<int>> oldIdQueue;
queue<vector<int64_t>> revOffsetQueue;
queue<vector<string>> revisionRowQueue;
queue<unordered_map<int, int>> actorCountQueue;

// Mutexes and Condition Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxOldId, mtxRevOffset, mtxRevisionRow, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvOldId, cvRevOffset, cvRevisionRow, cvOutput;

// Control Variables
int total_queries = 0;
int num_queries_done = 0;
bool no_more_queries = false;

// Assume you provide these external functions:
vector<int> KNNWithIndicesOnly(faiss::IndexHNSWFlat&, float*, int);
string getRowByIndex(const string&, const string&, int);
string getRowByOffset(const string&, int64_t);
vector<string> extractColumns(const string&, const vector<int>&);
int64_t getOffsetForRevId(int rev_id);

// Stage 1: Query Reader
void queryReaderThread(const string& queryFilename) {
    ifstream file(queryFilename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int k; ss >> k;
        vector<float> emb;
        float x;
        while (ss >> x) emb.push_back(x);

        if (!emb.empty()) {
            float* query_data = new float[emb.size()];
            copy(emb.begin(), emb.end(), query_data);
            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({k, query_data});
            total_queries++;
            cvQuery.notify_one();
        }
    }
    no_more_queries = true;
    cvQuery.notify_all();
}

// Stage 2: iKNN
void iKNNThread(faiss::IndexHNSWFlat& index) {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || no_more_queries; });
        if (queryQueue.empty() && no_more_queries) break;

        auto [k, query_data] = queryQueue.front();
        queryQueue.pop();
        lock.unlock();

        vector<int> rowIds = KNNWithIndicesOnly(index, query_data, k);
        delete[] query_data;

        {
            lock_guard<mutex> lock2(mtxRowId);
            rowIdQueue.push(rowIds);
            cvRowId.notify_one();
        }
    }
}

// Stage 3: Row Extractor
void rowExtractorThread(const string& csv, const string& offsets) {
    while (true) {
        unique_lock<mutex> lock(mtxRowId);
        cvRowId.wait(lock, [] { return !rowIdQueue.empty() || no_more_queries; });
        if (rowIdQueue.empty() && no_more_queries) break;

        auto rowIds = rowIdQueue.front();
        rowIdQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (int id : rowIds)
            rows.push_back(getRowByIndex(csv, offsets, id));

        {
            lock_guard<mutex> lock2(mtxRow);
            rowQueue.push(rows);
            cvRow.notify_one();
        }
    }
}

// Stage 4: Extract old_id
void columnExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !rowQueue.empty() || no_more_queries; });
        if (rowQueue.empty() && no_more_queries) break;

        auto rows = rowQueue.front();
        rowQueue.pop();
        lock.unlock();

        vector<int> oldIds;
        for (auto& row : rows)
            oldIds.push_back(stoi(extractColumns(row, {0})[0]));

        {
            lock_guard<mutex> lock2(mtxOldId);
            oldIdQueue.push(oldIds);
            cvOldId.notify_one();
        }
    }
}

// Stage 5: Get rev_id offsets
void revIdToOffsetThread() {
    while (true) {
        unique_lock<mutex> lock(mtxOldId);
        cvOldId.wait(lock, [] { return !oldIdQueue.empty() || no_more_queries; });
        if (oldIdQueue.empty() && no_more_queries) break;

        auto revIds = oldIdQueue.front();
        oldIdQueue.pop();
        lock.unlock();

        vector<int64_t> offsets;
        for (int id : revIds)
            offsets.push_back(getOffsetForRevId(id));

        {
            lock_guard<mutex> lock2(mtxRevOffset);
            revOffsetQueue.push(offsets);
            cvRevOffset.notify_one();
        }
    }
}

// Stage 6: Extract rev_actor
void revisionRowExtractorThread(const string& revisionFile) {
    while (true) {
        unique_lock<mutex> lock(mtxRevOffset);
        cvRevOffset.wait(lock, [] { return !revOffsetQueue.empty() || no_more_queries; });
        if (revOffsetQueue.empty() && no_more_queries) break;

        auto offsets = revOffsetQueue.front();
        revOffsetQueue.pop();
        lock.unlock();

        vector<string> rows;
        for (auto& off : offsets)
            rows.push_back(getRowByOffset(revisionFile, off));

        {
            lock_guard<mutex> lock2(mtxRevisionRow);
            revisionRowQueue.push(rows);
            cvRevisionRow.notify_one();
        }
    }
}

// Stage 7: Count rev_actor
void actorCounterThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRevisionRow);
        cvRevisionRow.wait(lock, [] { return !revisionRowQueue.empty() || no_more_queries; });
        if (revisionRowQueue.empty() && no_more_queries) break;

        auto rows = revisionRowQueue.front();
        revisionRowQueue.pop();
        lock.unlock();

        unordered_map<int, int> actorCount;
        for (auto& row : rows) {
            int actor = stoi(extractColumns(row, {2})[0]); // rev_actor = column 2
            actorCount[actor]++;
        }

        {
            lock_guard<mutex> lock2(mtxOutput);
            actorCountQueue.push(actorCount);
            cvOutput.notify_one();
        }
    }
}

// Stage 8: Output Writer
void outputWriterThread(const string& outputFile) {
    ofstream file(outputFile);
    while (true) {
        unique_lock<mutex> lock(mtxOutput);
        cvOutput.wait(lock, [] { return !actorCountQueue.empty() || no_more_queries; });
        if (actorCountQueue.empty() && no_more_queries) break;

        auto counter = actorCountQueue.front();
        actorCountQueue.pop();
        lock.unlock();

        vector<pair<int, int>> sorted(counter.begin(), counter.end());
        sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return b.second < a.second; });

        for (auto& [actor, count] : sorted)
            file << actor << "," << count << "\n";
    }
    file.close();
}

int main() {
    string queryFile = "q10_queries.txt";
    string indexFile = "old_text_embedding_index.faiss";
    string textFile = "text.csv";
    string textOffsets = "text_offsets.bin";
    string revisionFile = "revisions.csv";
    string outputFile = "q10_output.txt";

    faiss::IndexHNSWFlat* index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(indexFile));

    thread t1(queryReaderThread, queryFile);
    thread t2(iKNNThread, ref(*index));
    thread t3(rowExtractorThread, textFile, textOffsets);
    thread t4(columnExtractorThread);
    thread t5(revIdToOffsetThread);
    thread t6(revisionRowExtractorThread, revisionFile);
    thread t7(actorCounterThread);
    thread t8(outputWriterThread, outputFile);

    t1.join(); t2.join(); t3.join(); t4.join();
    t5.join(); t6.join(); t7.join(); t8.join();

    delete index;
    return 0;
}