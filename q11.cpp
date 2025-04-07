#include <bits/stdc++.h>
#include <faiss/IndexHNSW.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

using namespace std;

// Shared Queues for Each Stage
queue<pair<float, float*>> queryQueue;
queue<vector<int>> rowIdQueue;
queue<vector<string>> rowQueue;
queue<vector<int>> pageIdQueue;
queue<vector<int>> revIdQueue;
queue<vector<string>> revRowQueue;
queue<vector<pair<int, int>>> actorEditQueue;
queue<vector<pair<int, int>>> outputQueue;

// Synchronization Variables
mutex mtxQuery, mtxRowId, mtxRow, mtxPageId, mtxRevId, mtxRevRow, mtxActorEdit, mtxOutput;
condition_variable cvQuery, cvRowId, cvRow, cvPageId, cvRevId, cvRevRow, cvActorEdit, cvOutput;

// Control Variables
int num_queries_done = 0;
bool no_more_queries = false;
int total_queries = 0;

// Mocked Utility Functions
vector<float> KNNWithDistancesOnly(faiss::IndexHNSWFlat& index, float* query_data, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query_data, k, D.data(), I.data());
    return D;
}

pair<vector<int>, vector<float>> KNNWithIndicesAndDistances(faiss::IndexHNSWFlat& index, float* query_data, int k) {
    vector<faiss::idx_t> I(k);
    vector<float> D(k);
    index.search(1, query_data, k, D.data(), I.data());
    return {vector<int>(I.begin(), I.end()), D};
}

string getRowByIndex(const string& csvFilename, const string& offsetFilename, int rowId) {
    return "mocked_row"; // Replace with actual file logic
}

string getRowByRevID(const string& filename, const string& offsetFilename, int revId) {
    return "mocked_revision_row"; // Replace with actual file logic
}

vector<string> extractColumns(const string& row, const vector<int>& columnIndices) {
    return {"123", "1"}; // Replace with actual CSV parsing logic
}

unordered_map<int, vector<int>> revPageMap; // Precomputed: page_id -> list of rev_id
unordered_map<int, int> revOffsets;         // Precomputed: rev_id -> offset

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
        float d;
        ss >> d;

        vector<float> query_embedding;
        float value;
        while (ss >> value) query_embedding.push_back(value);

        if (!query_embedding.empty()) {
            float* query_data = new float[query_embedding.size()];
            copy(query_embedding.begin(), query_embedding.end(), query_data);

            lock_guard<mutex> lock(mtxQuery);
            queryQueue.push({d, query_data});
            total_queries++;
        }
        cvQuery.notify_one();
    }

    file.close();
    no_more_queries = true;
    cvQuery.notify_all();
}

// **iKNN Thread**
void iKNNThread(faiss::IndexHNSWFlat& index) {
    while (true) {
        unique_lock<mutex> lock(mtxQuery);
        cvQuery.wait(lock, [] { return !queryQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (queryQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        auto [d, query_data] = queryQueue.front();
        queryQueue.pop();
        lock.unlock();

        int k = 1;
        vector<float> distances;

        while (true) {
            distances = KNNWithDistancesOnly(index, query_data, k);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            if (it != distances.end()) {
                k = (it - distances.begin()) + 1;
                break;
            }
            k *= 2;
        }

        auto [indices, finalDistances] = KNNWithIndicesAndDistances(index, query_data, k);

        vector<int> rowIds;
        for (size_t i = 0; i < finalDistances.size(); i++) {
            if (finalDistances[i] < d && indices[i] != -1) {
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
void columnExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRow);
        cvRow.wait(lock, [] { return !rowQueue.empty() || (no_more_queries && num_queries_done == total_queries); });

        if (rowQueue.empty() && no_more_queries && num_queries_done == total_queries) break;

        vector<string> rows = rowQueue.front();
        rowQueue.pop();
        lock.unlock();

        vector<int> pageIds;
        for (const string& row : rows) {
            pageIds.push_back(stoi(extractColumns(row, {0})[0]));
        }

        lock_guard<mutex> lockPageId(mtxPageId);
        pageIdQueue.push(pageIds);
        cvPageId.notify_one();
    }
}

// **Rev ID Extractor Thread**
void revIdExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxPageId);
        cvPageId.wait(lock, [] { return !pageIdQueue.empty(); });

        vector<int> pageIds = pageIdQueue.front();
        pageIdQueue.pop();
        lock.unlock();

        vector<int> revIds;
        for (int pageId : pageIds) {
            auto& ids = revPageMap[pageId];
            revIds.insert(revIds.end(), ids.begin(), ids.end());
        }

        lock_guard<mutex> lockRev(mtxRevId);
        revIdQueue.push(revIds);
        cvRevId.notify_one();
    }
}

// **Revision Row Extractor Thread**
void revRowExtractorThread(const string& revFilename, const string& revOffsetFilename) {
    while (true) {
        unique_lock<mutex> lock(mtxRevId);
        cvRevId.wait(lock, [] { return !revIdQueue.empty(); });

        vector<int> revIds = revIdQueue.front();
        revIdQueue.pop();
        lock.unlock();

        vector<string> revRows;
        for (int revId : revIds) {
            revRows.push_back(getRowByRevID(revFilename, revOffsetFilename, revId));
        }

        lock_guard<mutex> lockRevRow(mtxRevRow);
        revRowQueue.push(revRows);
        cvRevRow.notify_one();
    }
}

// **Column Extractor for rev_actor and rev_minor_edit**
void actorEditExtractorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxRevRow);
        cvRevRow.wait(lock, [] { return !revRowQueue.empty(); });

        vector<string> rows = revRowQueue.front();
        revRowQueue.pop();
        lock.unlock();

        vector<pair<int, int>> actorEdits;
        for (const string& row : rows) {
            auto cols = extractColumns(row, {0, 1}); // rev_actor, rev_minor_edit
            actorEdits.emplace_back(stoi(cols[0]), stoi(cols[1]));
        }

        lock_guard<mutex> lockActor(mtxActorEdit);
        actorEditQueue.push(actorEdits);
        cvActorEdit.notify_one();
    }
}

// **Aggregator Thread**
void actorAggregatorThread() {
    while (true) {
        unique_lock<mutex> lock(mtxActorEdit);
        cvActorEdit.wait(lock, [] { return !actorEditQueue.empty(); });

        vector<pair<int, int>> edits = actorEditQueue.front();
        actorEditQueue.pop();
        lock.unlock();

        unordered_map<int, int> countMap;
        for (const auto& [actor, minorEdit] : edits) {
            countMap[actor] += minorEdit;
        }

        vector<pair<int, int>> sorted(countMap.begin(), countMap.end());
        sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

        lock_guard<mutex> lockOutput(mtxOutput);
        outputQueue.push(sorted);
        cvOutput.notify_one();
    }
}

// **Output Writer Thread**
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

        vector<pair<int, int>> results = outputQueue.front();
        outputQueue.pop();
        lock.unlock();

        for (auto& [actor, count] : results) {
            file << actor << "," << count << "\n";
        }
    }

    file.close();
}

int main() {
    string queryFilename = "q11_queries.txt";
    string indexFilename = "page_embedding_index.faiss";
    string csvFilename = "page.csv";
    string offsetFilename = "page_offsets.bin";
    string revisionFilename = "revisions.csv";
    string revisionOffsetFilename = "revision_offsets.bin";
    string outputFilename = "q11_output.txt";

    faiss::IndexHNSWFlat* index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(indexFilename));

    thread t1(queryReaderThread, queryFilename);
    thread t2(iKNNThread, ref(*index));
    thread t3(rowExtractorThread, csvFilename, offsetFilename);
    thread t4(columnExtractorThread);
    thread t5(revIdExtractorThread);
    thread t6(revRowExtractorThread, revisionFilename, revisionOffsetFilename);
    thread t7(actorEditExtractorThread);
    thread t8(actorAggregatorThread);
    thread t9(outputWriterThread, outputFilename);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();
    t9.join();

    delete index;
    return 0;
}
