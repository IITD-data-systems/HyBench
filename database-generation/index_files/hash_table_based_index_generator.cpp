#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

// Reusing your index functions
void build_index(const std::vector<int>& index_ids,
                 std::unordered_map<int, std::vector<int>>& index) {
    for (int row_id = 0; row_id < index_ids.size(); ++row_id) {
        index[index_ids[row_id]].push_back(row_id);
    }
}

void save_index(const std::unordered_map<int, std::vector<int>>& index,
                const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    size_t map_size = index.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for (const auto& [key, vec] : index) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));
        size_t vec_size = vec.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        out.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(int));
    }
    out.close();
}

// New: Reads the 2nd column from CSV and returns rev_pages
std::vector<int> read_rev_pages_from_csv(const std::string& filename,int column_number) {
    std::ifstream file(filename);
    std::vector<int> index_ids;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string col;
        int col_index = 0;
        int index_id = -1;

        while (std::getline(ss, col, ',')) {
            if (col_index == column_number) { 
                index_id = std::stoi(col);
                break;
            }
            ++col_index;
        }

        if (index_id != -1){
            index_ids.push_back(index_id);}
        else{
            cout<< "Row Number" << index_ids.size() << "doesn't have enough columns"<<endl;
            return index_ids;

        }

    }

    return index_ids;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./index_builder <input.csv> <output_index_file> <column_number>\n";
        return 1;
    }

    std::string csv_filename = argv[1];
    std::string index_filename = argv[2];
    int column_number = stoi(argv[3]);

    // Step 1: Read rev_pages from CSV
    std::vector<int> rev_pages = read_rev_pages_from_csv(csv_filename,column_number);

    // Step 2: Build index
    std::unordered_map<int, std::vector<int>> index;
    build_index(rev_pages, index);

    // Step 3: Save index
    save_index(index, index_filename);

    std::cout << "Index built and saved to " << index_filename << "\n";
    return 0;
}
