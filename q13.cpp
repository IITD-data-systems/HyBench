#include <bits/stdc++.h>
#include <faiss/IndexFlat.h>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

const int DIM = 128; // embedding dimension

// Stage 1 → Stage 2
queue<pair<int, vector<float>>> embeddingQueue;
mutex mtx1;
condition_variable cv1;

// Stage 2 → Stage 3 and Stage 4
unordered_map<int, vector<pair<vector<float>, int>>> clToEmbeddings;
mutex mtx2;
bool stage2Done = false;
condition_variable cv2;

// Stage 3 → Stage 4
unordered_map<int, vector<float>> clToAvgEmbedding;
mutex mtx3;
bool stage3Done = false;
condition_variable cv3;

// Stage 4 → Stage 5
unordered_map<int, int> finalMap; // cl_to -> page_id
mutex mtx4;
bool stage4Done = false;
condition_variable cv4;

// ---------- Mocked Helper ----------
int get_cl_to_for_page(int page_id) {
    return page_id % 10; // Dummy function
}

// ---------- Stage 1 ----------
void stage1(const string& filename) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        getline(ss, token, ',');
        int page_id = stoi(token);
        vector<float> emb;
        while (getline(ss, token, ',')) {
            emb.push_back(stof(token));
        }
        {
            lock_guard<mutex> lk(mtx1);
            embeddingQueue.push({page_id, emb});
        }
        cv1.notify_one();
    }
    {
        lock_guard<mutex> lk(mtx1);
        embeddingQueue.push({-1, {}}); // EOF marker
    }
    cv1.notify_one();
}

// ---------- Stage 2 ----------
void stage2() {
    while (true) {
        unique_lock<mutex> lk(mtx1);
        cv1.wait(lk, [] { return !embeddingQueue.empty(); });
        auto [page_id, emb] = embeddingQueue.front(); embeddingQueue.pop();
        lk.unlock();
        if (page_id == -1) break;

        int cl_to = get_cl_to_for_page(page_id);

        {
            lock_guard<mutex> lk2(mtx2);
            clToEmbeddings[cl_to].push_back({emb, page_id});
        }
    }
    stage2Done = true;
    cv2.notify_all();
}

// ---------- Stage 3 ----------
void stage3() {
    unique_lock<mutex> lk(mtx2);
    cv2.wait(lk, [] { return stage2Done; });

    for (const auto& [cl, vecs] : clToEmbeddings) {
        vector<float> sum(DIM, 0.0f);
        for (const auto& [emb, _] : vecs) {
            for (int i = 0; i < DIM; ++i) sum[i] += emb[i];
        }
        for (float& x : sum) x /= vecs.size();
        clToAvgEmbedding[cl] = sum;
    }
    stage3Done = true;
    cv3.notify_all();
}

// ---------- Stage 4 ----------
void stage4() {
    unique_lock<mutex> lk3(mtx3);
    cv3.wait(lk3, [] { return stage3Done; });
    lk3.unlock();

    for (const auto& [cl, avg_emb] : clToAvgEmbedding) {
        faiss::IndexFlatL2 index(DIM);
        vector<int> ids;
        for (const auto& [emb, pid] : clToEmbeddings[cl]) {
            index.add(1, emb.data());
            ids.push_back(pid);
        }

        vector<faiss::idx_t> idx(1);
        vector<float> dist(1);
        index.search(1, avg_emb.data(), 1, dist.data(), idx.data());

        finalMap[cl] = ids[idx[0]];
    }
    stage4Done = true;
    cv4.notify_all();
}

// ---------- Stage 5 ----------
void stage5(const string& outputFile) {
    unique_lock<mutex> lk(mtx4);
    cv4.wait(lk, [] { return stage4Done; });
    ofstream out(outputFile);
    for (auto& [cl, pid] : finalMap) {
        out << cl << "," << pid << "\n";
    }
    out.close();
}

// ---------- Main ----------
int main() {
    thread t1(stage1, "page.csv");
    thread t2(stage2);
    thread t3(stage3);
    thread t4(stage4);
    thread t5(stage5, "output.csv");

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    return 0;
}
