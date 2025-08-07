#ifndef PIPELINE_STAGES_CPP_INCLUDED
#define PIPELINE_STAGES_CPP_INCLUDED

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>  
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <faiss/IndexHNSW.h>
#include <utility>
#include <stdexcept>

using namespace std;
class MyIndex {
public:
    virtual ~MyIndex() = default;

    virtual void add(float* data, int n) = 0;

    virtual std::pair<std::vector<faiss::idx_t>, std::vector<float>>
    KNNWithIndicesAndDistances(float* query, int k) = 0;

    virtual std::vector<faiss::idx_t>
    KNNWithIndicesOnly(float* query, int k) = 0;

    virtual std::vector<float>
    KNNWithDistancesOnly(float* query, int k) = 0;
    virtual vector<long int> KNNWithDistanceUp(float* query_data, float d, int k=1)=0;
    virtual vector<long int> KNNWithDistanceUpDownLimit(float* query_data, float d,float d_star, int k)=0;
    virtual vector<long int> KNNWithDistanceUpDown(float* query_data, float d,float d_star)=0;
    virtual pair<vector<long int>,vector<float>> KNNWithDistanceUpDownWithDistances(float* query_data, float d,float d_star)=0;
    virtual void set_search_parameter(int value) = 0;
    virtual std::string index_kind() const = 0;
    virtual std::string metric_type() const = 0;
    virtual void get_stats() = 0;
};

// -------------------- IVFFlatIndex --------------------

class IVFFlatIndex : public MyIndex {
private:
    faiss::IndexIVFFlat* index;

public:
    IVFFlatIndex(const std::string& filename) {
        faiss::Index* base_index = faiss::read_index(filename.c_str());
        index = dynamic_cast<faiss::IndexIVFFlat*>(base_index);
        if (!index) {
            throw std::runtime_error("Failed to load IndexIVFFlat from file: " + filename);
        }
    }

    ~IVFFlatIndex() override {
        delete index;
    }

    void set_search_parameter(int value) override {
        index->nprobe = value;
        std::cout << "Set nprobe = " << value << " for IVFFlatIndex\n";
    }
    
    void add(float* data, int n) override {
        index->add(n, data);
    }

    std::pair<std::vector<faiss::idx_t>, std::vector<float>>
    KNNWithIndicesAndDistances(float* query, int k) override {
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);
        index->search(1, query, k, distances.data(), indices.data());
        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
            for (auto& d : distances) {
                d = 1.0f - d;  // convert similarity to cosine distance
            }
        }    
        return {indices, distances};
    }

    std::vector<faiss::idx_t>
    KNNWithIndicesOnly(float* query, int k) override {
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);

        
        index->search(1, query, k, distances.data(), indices.data());
        return indices;
        
    }

    std::vector<float>
    KNNWithDistancesOnly(float* query, int k) override {
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> dummy_indices(k);
        index->search(1, query, k, distances.data(), dummy_indices.data());
        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
            for (auto& d : distances) {
                d = 1.0f - d;  // convert similarity to cosine distance
            }
        }    
        return distances;
    }

    faiss::IndexIVFFlat* getRawIndex() {
        return index;
    }
    std::string index_kind() const override {
        return "ivfflat";
    }

    std::string metric_type() const override {
        if (index->metric_type == faiss::METRIC_L2) return "l2";
        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) return "cos";
        return "unknown";
    }
    void get_stats() override {
    printf("IVFFlat index:\n");
    printf("  Number of lists (nlist): %ld\n", index->nlist);
    printf("  nprobe (search lists): %ld\n", index->nprobe);
    printf("  Metric type: %d\n", index->metric_type);
    printf("  Vectors indexed: %ld\n", index->ntotal);

    // Try to get quantizer info
    auto* ivf_index = dynamic_cast<faiss::IndexIVF*>(index);
    if (ivf_index && ivf_index->quantizer) {
        printf("  Quantizer type: %s\n", typeid(*ivf_index->quantizer).name());
        printf("  Quantizer metric type: %d\n", ivf_index->quantizer->metric_type);

        // Optional: check known quantizer types
        if (dynamic_cast<faiss::IndexFlatL2*>(ivf_index->quantizer)) {
            printf("    (This is IndexFlatL2)\n");
        } else if (dynamic_cast<faiss::IndexFlatIP*>(ivf_index->quantizer)) {
            printf("    (This is IndexFlatIP)\n");
        }
    } else {
        printf("  Quantizer: not available or not an IVF index.\n");
    }
}

    vector<long int> KNNWithDistanceUp(float* query_data, float d, int k=1) override {
    bool is_inner_product = (index->metric_type == faiss::METRIC_INNER_PRODUCT);
    

    float range_threshold = d;

    if (is_inner_product) {
        
        range_threshold = 1.0f - d;  
    } 

    // Perform range search
    faiss::RangeSearchResult res(1);
    index->range_search(1, query_data, range_threshold, &res);

    struct Candidate {
        long int label;
        float distance;  // Unified distance (L2 or cosine distance)
    };

    vector<Candidate> candidates;
    candidates.reserve(res.lims[1] - res.lims[0]);

    for (size_t i = res.lims[0]; i < res.lims[1]; i++) {
        if (res.labels[i] == -1) continue;

        float distance = res.distances[i];

        if (is_inner_product) {
            // Convert inner product to cosine distance
            distance = 1.0f - distance;
        } 

        if (distance < d) {
            candidates.push_back({res.labels[i], distance});
        }
    }

    // Sort by distance first, then label as tie-breaker
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.distance != b.distance)
                      return a.distance < b.distance;
                  else
                      return a.label < b.label;
              });

    // Extract sorted labels
    vector<long int> indices;
    indices.reserve(candidates.size());
    for (const auto& c : candidates) {
        indices.push_back(c.label);
    }

    return indices;
    }


    vector<long int> KNNWithDistanceUpDownLimit(float* query_data, float d, float d_star, int k) override {
    // Step 1: Determine the correct threshold to use for range_search
    float range_threshold = d_star;
    bool is_inner_product = (index->metric_type == faiss::METRIC_INNER_PRODUCT);

    if (is_inner_product) {
        // Convert cosine distance to inner product
        // d, d_star are cosine distances: 0 = identical, 2 = opposite
        range_threshold = 1.0f - d_star;  
    } 

    // Step 2: Perform range search
    faiss::RangeSearchResult res(1);
    index->range_search(1, query_data, range_threshold, &res);

    struct Candidate {
        long int label;
        float distance;  // Unified distance (L2 or cosine distance)
    };

    vector<Candidate> candidates;

    // Step 3: Filter and compute distances consistently
    for (size_t i = res.lims[0]; i < res.lims[1]; i++) {
        if (res.labels[i] == -1) continue;

        float distance = res.distances[i];
        

        if (is_inner_product) {
            distance = 1.0f - distance;  // Convert inner product to cosine distance
        } 

        if (distance >= d && distance <= d_star) {
            candidates.push_back({res.labels[i], distance});
        }
    }

    // Step 4: Sort by distance then label
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.distance != b.distance)
                      return a.distance < b.distance;
                  else
                      return a.label < b.label;
              });

    // Step 5: Return top k labels
    vector<long int> final_indices;
    for (size_t i = 0; i < std::min((size_t)k, candidates.size()); i++) {
        final_indices.push_back(candidates[i].label);
    }

    return final_indices;
}

vector<long int> KNNWithDistanceUpDown(float* query_data, float d, float d_star) override {
    // Step 1: Determine the correct threshold to use for range_search
    float range_threshold = d_star;
    bool is_inner_product = (index->metric_type == faiss::METRIC_INNER_PRODUCT);
    

    if (is_inner_product) {
        // Convert cosine distance to inner product
        // d, d_star are cosine distances: 0 = identical, 2 = opposite
        range_threshold = 1.0f - d_star;  
    } 

    // Step 2: Perform range search
    faiss::RangeSearchResult res(1);
    index->range_search(1, query_data, range_threshold, &res);

    struct Candidate {
        long int label;
        float distance;  // Unified distance (L2 or cosine distance)
    };

    vector<Candidate> candidates;

    // Step 3: Filter and compute distances consistently
    for (size_t i = res.lims[0]; i < res.lims[1]; i++) {
        if (res.labels[i] == -1) continue;

        float distance = res.distances[i];
        

        if (is_inner_product) {
            distance = 1.0f - distance;  // Convert inner product to cosine distance
        } 

        if (distance >= d && distance <= d_star) {
            candidates.push_back({res.labels[i], distance});
        }
    }

    // Step 4: Sort by distance then label
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.distance != b.distance)
                      return a.distance < b.distance;
                  else
                      return a.label < b.label;
              });

    // Step 5: Return top k labels
    vector<long int> final_indices;
    for (size_t i = 0; i < candidates.size(); i++) {
        final_indices.push_back(candidates[i].label);
    }

    return final_indices;
}

pair<vector<long int>,vector<float>> KNNWithDistanceUpDownWithDistances(float* query_data, float d, float d_star) override {
    // Step 1: Determine the correct threshold to use for range_search
    float range_threshold = d_star;
    bool is_inner_product = (index->metric_type == faiss::METRIC_INNER_PRODUCT);
    

    if (is_inner_product) {
        // Convert cosine distance to inner product
        // d, d_star are cosine distances: 0 = identical, 2 = opposite
        range_threshold = 1.0f - d_star;  
    } 

    // Step 2: Perform range search
    faiss::RangeSearchResult res(1);
    index->range_search(1, query_data, range_threshold, &res);

    struct Candidate {
        long int label;
        float distance;  // Unified distance (L2 or cosine distance)
    };

    vector<Candidate> candidates;

    // Step 3: Filter and compute distances consistently
    for (size_t i = res.lims[0]; i < res.lims[1]; i++) {
        if (res.labels[i] == -1) continue;

        float distance = res.distances[i];
        

        if (is_inner_product) {
            distance = 1.0f - distance;  // Convert inner product to cosine distance
        } 

        if (distance >= d && distance <= d_star) {
            candidates.push_back({res.labels[i], distance});
        }
    }

    // Step 4: Sort by distance then label
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.distance != b.distance)
                      return a.distance < b.distance;
                  else
                      return a.label < b.label;
              });

    // Step 5: Return top k labels
    vector<long int> final_indices;
    vector<float> final_distances;
    for (size_t i = 0; i < candidates.size(); i++) {
        final_indices.push_back(candidates[i].label);
        final_distances.push_back(candidates[i].distance);
    }

    return {final_indices,final_distances};
}

};

// -------------------- HNSWIndex --------------------

class HNSWIndex : public MyIndex {
private:
    faiss::IndexHNSWFlat* index;

public:
    HNSWIndex(const std::string& filename) {
        faiss::Index* base_index = faiss::read_index(filename.c_str());
        index = dynamic_cast<faiss::IndexHNSWFlat*>(base_index);
        if (!index) {
            throw std::runtime_error("Failed to load IndexHNSWFlat from file: " + filename);
        }
    }

    ~HNSWIndex() override {
        delete index;
    }

    void set_search_parameter(int value) override {
        index->hnsw.efSearch = value;
        std::cout << "Set efSearch = " << value << " for HNSWIndex\n";
    }

    void add(float* data, int n) override {
        index->add(n, data);
    }
    void get_stats() override{
        return;
    }

    std::pair<std::vector<faiss::idx_t>, std::vector<float>>
    KNNWithIndicesAndDistances(float* query, int k) override {
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);
        index->search(1, query, k, distances.data(), indices.data());

        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
            // recompute distances by reconstructing vectors
            std::vector<float> vec(index->d);
            for (int i = 0; i < k; i++) {
                if (indices[i] < 0) {  // invalid index
                    distances[i] = std::numeric_limits<float>::max();
                    continue;
                }
                index->reconstruct(indices[i], vec.data());
                
                float ip = 0.0f;
                for (int j = 0; j < index->d; j++) {
                    ip += query[j] * vec[j];
                }
                distances[i] = 1.0f - ip;
            }
        }

        

        return {indices, distances};
    }

    std::vector<faiss::idx_t>
    KNNWithIndicesOnly(float* query, int k) override {
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);
        index->search(1, query, k, distances.data(), indices.data());
             
        return indices;
    }

    std::vector<float>
    KNNWithDistancesOnly(float* query, int k) override {
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> dummy_indices(k);
        index->search(1, query, k, distances.data(), dummy_indices.data());

        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
            std::vector<float> vec(index->d);
            for (int i = 0; i < k; i++) {
                if (dummy_indices[i] < 0) {
                    distances[i] = std::numeric_limits<float>::max();
                    continue;
                }
                index->reconstruct(dummy_indices[i], vec.data());
                float ip = 0.0f;
                for (int j = 0; j < index->d; j++) {
                    ip += query[j] * vec[j];
                }
                distances[i] = 1.0f - ip;
            }
        }
        
        cout<<endl;

        
        

        return distances;
    }

    faiss::IndexHNSWFlat* getRawIndex() {
        return index;
    }
    std::string index_kind() const override {
        return "hnsw";
    }

    std::string metric_type() const override {
        if (index->metric_type == faiss::METRIC_L2) return "l2";
        if (index->metric_type == faiss::METRIC_INNER_PRODUCT) return "cos";
        return "unknown";
    }
    vector<long int> KNNWithDistanceUp(float* query_data, float d, int k=1) override {
        int optimalK = 1;
        vector<float> distances;

        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            if (it != distances.end()) {
                optimalK = (it - distances.begin());
                break;
            }
            optimalK *= 2;
            
        }

        auto indices = KNNWithIndicesOnly(query_data, optimalK);
        
        vector<long int> final_indices;
        for (int i = 0; i < optimalK; i++) {
            if(indices[i]==-1)continue;
            
            final_indices.push_back(indices[i]);
            
        }
        return final_indices;
    }

    vector<long int> KNNWithDistanceUpDownLimit(float* query_data , float d, float d_star, int k) override{
        int optimalK = k;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()) + 1; // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK+k-1);
        
        // **Step 3: Filtering Indices Between d and d***

        vector<long int> final_indices;
        for (size_t i = optimalK-1; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                
            }
        }
        return final_indices;
    }

    vector<long int> KNNWithDistanceUpDown(float* query_data , float d, float d_star) override{
        int optimalK = 10;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = upper_bound(distances.begin(), distances.end(), d_star);
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()); // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK);
        
        // **Step 3: Filtering Indices Between d and d***
        int start= lower_bound(finalDistances.begin(), finalDistances.end(), d) - finalDistances.begin();
        vector<long int> final_indices;
        for (size_t i = start; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                
            }
        }
        return final_indices;
    }

    pair<vector<long int>,vector<float>> KNNWithDistanceUpDownWithDistances(float* query_data , float d, float d_star) override{
        int optimalK = 10;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = upper_bound(distances.begin(), distances.end(), d_star);
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()); // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK);
        
        // **Step 3: Filtering Indices Between d and d***
        int start= lower_bound(finalDistances.begin(), finalDistances.end(), d) - finalDistances.begin();
        vector<long int> final_indices;
        vector<float> final_distances;
        for (size_t i = start; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                final_distances.push_back(finalDistances[i]);
                
            }
        }
        return {final_indices,final_distances};
    }

};


// HNSW using hnswlib-------------------------------------------------------------------------------------------------------

#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "../database-generation/index_files/hnswlib/hnswlib/hnswlib.h" // Adjust include path as needed

class HNSWLibIndex : public MyIndex {
private:
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    int dim;
    std::string metric;

public:
    // Constructor
    HNSWLibIndex(const std::string& filename, const std::string& metric_ = "l2", int dim_=384)
    : dim(dim_), metric(metric_)
{
    if (metric == "l2") {
        space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (metric == "cos") {
        space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
        throw std::runtime_error("Unsupported metric type: " + metric);
    }

    // Idiomatic: Load index directly using the constructor
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), filename);
}


    ~HNSWLibIndex() override = default;

    void set_search_parameter(int ef) override {
        index->setEf(ef);
        std::cout << "Set ef = " << ef << " for HNSWLibIndex\n";
    }

    void add(float* data, int n) override {
        for (int i = 0; i < n; i++) {
            index->addPoint(data + i * dim, static_cast<size_t>(i));
        }
    }

    std::pair<std::vector<long int>, std::vector<float>> KNNWithIndicesAndDistances(float* query, int k) override {
        auto result = index->searchKnn(query, k);
        std::vector<long int> indices;
        std::vector<float> distances;
        while (!result.empty()) {
            const auto& res = result.top();
            indices.push_back(static_cast<long int>(res.second));
            
            distances.push_back(res.first);
            result.pop();
        }
        
        std::reverse(indices.begin(), indices.end());
        std::reverse(distances.begin(), distances.end());
        
        
        return {indices, distances};
    }

    void get_stats() override{
        return;
    }
    std::vector<long int> KNNWithIndicesOnly(float* query, int k) override {
        auto result = index->searchKnn(query, k);
        std::vector<long int> indices;
        while (!result.empty()) {
            indices.push_back(static_cast<long int>(result.top().second));
            result.pop();
        }
        std::reverse(indices.begin(), indices.end());
        return indices;
    }

    std::vector<float> KNNWithDistancesOnly(float* query, int k) override {
        auto result = index->searchKnn(query, k);
        std::vector<float> distances;
        while (!result.empty()) {
            
            distances.push_back(result.top().first);

            result.pop();
        }
        
        std::reverse(distances.begin(), distances.end());
        return distances;
    }

    std::string index_kind() const override {
        return "hnswlib";
    }

    std::string metric_type() const override {
        return metric;
    }

    vector<long int> KNNWithDistanceUp(float* query_data, float d, int k=1) override {
        int optimalK = k;
        vector<float> distances;

        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            if (it != distances.end()) {
                optimalK = (it - distances.begin());
                break;
            }
            optimalK *= 2;
            
        }

        auto indices = KNNWithIndicesOnly(query_data, optimalK);
        
        vector<long int> final_indices;
        for (int i = 0; i < optimalK; i++) {
            if(indices[i]==-1)continue;
            
            final_indices.push_back(indices[i]);
            
        }
        return final_indices;
    }

    vector<long int> KNNWithDistanceUpDownLimit(float* query_data , float d, float d_star, int k)override {
        int optimalK = k;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = lower_bound(distances.begin(), distances.end(), d);
            
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()) + 1; // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK+k-1);
        
        // **Step 3: Filtering Indices Between d and d***
        
        vector<long int> final_indices;
        
        
        for (size_t i = optimalK-1; i < finalDistances.size(); i++) {
            
            
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                
                
            }
        }
        return final_indices;
    }
    vector<long int> KNNWithDistanceUpDown(float* query_data , float d, float d_star) override{
        int optimalK = 10;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = upper_bound(distances.begin(), distances.end(), d_star);
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()); // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK);
        
        // **Step 3: Filtering Indices Between d and d***
        int start= lower_bound(finalDistances.begin(), finalDistances.end(), d) - finalDistances.begin();
        vector<long int> final_indices;
        for (size_t i = start; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                
            }
        }
        return final_indices;
    }
    pair<vector<long int>,vector<float>> KNNWithDistanceUpDownWithDistances(float* query_data , float d, float d_star) override{
        int optimalK = 10;
        vector<float> distances;

        // **Step 1: Exponential k Growth**
        while (true) {
            distances = KNNWithDistancesOnly(query_data, optimalK);
            auto it = upper_bound(distances.begin(), distances.end(), d_star);
            
            if (it != distances.end()) {
                optimalK = (it - distances.begin()); // Set optimalK to first index where distance >= d
                break;
            }
            
            optimalK *= 2;
        }


        // **Step 2: Final kNN Call (Using New KNN Function)**
        auto [indices, finalDistances] = KNNWithIndicesAndDistances(query_data, optimalK);
        
        // **Step 3: Filtering Indices Between d and d***
        int start= lower_bound(finalDistances.begin(), finalDistances.end(), d) - finalDistances.begin();
        vector<long int> final_indices;
        vector<float> final_distances;
        for (size_t i = start; i < finalDistances.size(); i++) {
            if (finalDistances[i] <= d_star && indices[i]!=-1) {
                
                final_indices.push_back(indices[i]);
                final_distances.push_back(finalDistances[i]);
                
            }
        }
        return {final_indices,final_distances};
    }
};



void saveVectorOfVectors(const vector<vector<float>>& data, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out) throw runtime_error("Cannot open file for writing.");

    size_t outerSize = data.size();
    out.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

    for (const auto& inner : data) {
        size_t innerSize = inner.size();
        out.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
        out.write(reinterpret_cast<const char*>(inner.data()), innerSize * sizeof(float));
    }

    out.close();
}

vector<vector<float>> loadVectorOfVectors(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) throw runtime_error("Cannot open file for reading.");

    size_t outerSize;
    in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));

    vector<vector<float>> data(outerSize);
    for (size_t i = 0; i < outerSize; ++i) {
        size_t innerSize;
        in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));

        data[i].resize(innerSize);
        in.read(reinterpret_cast<char*>(data[i].data()), innerSize * sizeof(float));
    }

    in.close();
    return data;
}

void load_index(std::unordered_map<int, std::vector<int>>& index,
                const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    for (size_t i = 0; i < map_size; ++i) {
        int key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        index[key] = std::move(vec);
    }
    in.close();
}






std::string getRowByIndex(std::ifstream& csvFile, std::ifstream& offsetFile, uint64_t rowIndex) {
    
    if (!csvFile.is_open() || !offsetFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return "";
    }

    // Seek to the correct offset in the binary file
    offsetFile.seekg(rowIndex * sizeof(uint64_t), std::ios::beg);
    uint64_t offset;
    offsetFile.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    if (offsetFile.gcount() != sizeof(uint64_t)) {
        std::cerr << "Invalid row index!" << std::endl;
        return "";
    }

    // Seek to the row in CSV and read the line
    csvFile.seekg(offset, std::ios::beg);
    std::string row;
    std::getline(csvFile, row);

    return row;
}

std::vector<std::string> extractColumns(const std::string& row, const std::vector<int>& columnIndices) {
    std::vector<std::string> extractedColumns;
    std::vector<std::string> allColumns;

    std::string token;
    bool inQuotes = false;
    
    for (size_t i = 0; i < row.size(); ++i) {
        char c = row[i];
        if (inQuotes) {
            if (c == '"' && i + 1 < row.size() && row[i + 1] == '"') {
                token += '"';  // Escaped quote
                ++i;
            } else if (c == '"') {
                inQuotes = false;
            } else {
                token += c;
            }
        } else {
            if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                
                allColumns.push_back(token);
                token.clear();
            } else {
                token += c;
            }
        }
    }
    
    allColumns.push_back(token); // Last field

    // Extract only the requested columns
    for (int colIndex : columnIndices) {
        if (colIndex < static_cast<int>(allColumns.size())) {
            extractedColumns.push_back(allColumns[colIndex]);
        } else {
            extractedColumns.push_back(""); // Handle missing columns
        }
    }

    return extractedColumns;
}

int get_page_len(int index,std::ifstream& csvFile, std::ifstream& offsetFile){
    auto row=getRowByIndex(csvFile, offsetFile, index);
    return stoi(extractColumns(row, {0})[0]);
}

bool range_lie_identifier(float distance, vector<pair<float,float>>& distance_ranges){
    for(auto& [x,y] : distance_ranges){
        if(x<=distance && distance<=y)return true;
    }
    return false;
}

#endif // PIPELINE_STAGES_CPP_INCLUDED
