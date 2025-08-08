#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <stdexcept>
#include <unordered_map>


using namespace std;

// -------------------------------------------------------
// Load index from binary file
// -------------------------------------------------------
void load_index(std::unordered_map<int, std::vector<int>>& index,
                const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw runtime_error("Cannot open index file: " + filename);
    }
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

// -------------------------------------------------------
// Get CSV row by index using offset file
// -------------------------------------------------------
std::string getRowByIndex(std::ifstream& csvFile, std::ifstream& offsetFile, uint64_t rowIndex) {
    if (!csvFile.is_open() || !offsetFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return "";
    }

    offsetFile.seekg(rowIndex * sizeof(uint64_t), std::ios::beg);
    uint64_t offset;
    offsetFile.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    if (offsetFile.gcount() != sizeof(uint64_t)) {
        std::cerr << "Invalid row index!" << std::endl;
        return "";
    }

    csvFile.seekg(offset, std::ios::beg);
    std::string row;
    std::getline(csvFile, row);
    return row;
}

// -------------------------------------------------------
// Extract selected columns from CSV row
// -------------------------------------------------------
std::vector<std::string> extractColumns(const std::string& row, const std::vector<int>& columnIndices) {
    std::vector<std::string> extractedColumns;
    std::vector<std::string> allColumns;

    std::string token;
    bool inQuotes = false;
    
    for (size_t i = 0; i < row.size(); ++i) {
        char c = row[i];
        if (inQuotes) {
            if (c == '"' && i + 1 < row.size() && row[i + 1] == '"') {
                token += '"';
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

    for (int colIndex : columnIndices) {
        if (colIndex < static_cast<int>(allColumns.size())) {
            extractedColumns.push_back(allColumns[colIndex]);
        } else {
            extractedColumns.push_back("");
        }
    }
    return extractedColumns;
}

// -------------------------------------------------------
// Main
// -------------------------------------------------------
signed main(int argc, char* argv[]) {
    
    int scale_factor = stoll(argv[1]);
    string scale_factor_string = argv[1];

    string rev_page_index_filename = "index_files/rev_page_index.bin";
    std::unordered_map<int, std::vector<int>> rev_page_index;
    load_index(rev_page_index, rev_page_index_filename);

    string old_id_index_filename = "index_files/old_id_index.bin";
    unordered_map<int, vector<int>> old_id_index;
    load_index(old_id_index, old_id_index_filename);

    // // File names
    string page_file_name = "data_csv_files/page_csv_files/page.csv";
    string page_extra_file_name = "data_csv_files/page_csv_files/page_extra.csv";
    string revision_file_name = "data_csv_files/revision_csv_files/revision_clean.csv";
    string text_file_name = "data_csv_files/text_csv_files/text.csv";
    string revision_offset_file_name = "offsets_files/revision_offsets.bin";
    string text_offset_file_name = "offsets_files/text_offsets.bin";

    string page_file_name_new = "data_csv_files/page_csv_files/page_"+scale_factor_string+".csv";
    string page_extra_file_name_new = "data_csv_files/page_csv_files/page_extra_"+scale_factor_string+".csv";
    string revision_file_name_new = "data_csv_files/revision_csv_files/revision_clean_"+scale_factor_string+".csv";
    string text_file_name_new = "data_csv_files/text_csv_files/text_"+scale_factor_string+".csv";

    // // Open input files
    ifstream pageFile(page_file_name);
    ifstream pageExtraFile(page_extra_file_name);
    ifstream revFile(revision_file_name);
    ifstream textFile(text_file_name);
    ifstream revOffsetFile(revision_offset_file_name, ios::binary);
    ifstream textOffsetFile(text_offset_file_name, ios::binary);

    // // Open output files
    ofstream pageFileNew(page_file_name_new);
    ofstream pageExtraFileNew(page_extra_file_name_new);
    ofstream revFileNew(revision_file_name_new);
    ofstream textFileNew(text_file_name_new);

    long long cur_page_id = 1;
    long long cur_old_id = 1;

    string pageLine, pageExtraLine;
    while (getline(pageFile, pageLine) && getline(pageExtraFile, pageExtraLine)) {
        if (pageLine.empty()) continue;

        vector<string> pageCols = extractColumns(pageLine, {0, 1});

        int orig_page_id = stoll(pageCols[0]);
        string page_title = pageCols[1];

        // Replicate page & page_extra rows
        vector<int> new_page_ids;
        for (int i = 0; i < scale_factor; i++) {
            // page.csv
            pageFileNew << cur_page_id << "," << page_title << "\n";
            pageExtraFileNew << pageExtraLine << "\n";
            new_page_ids.push_back(cur_page_id);
            cur_page_id++;
        }

        // Get revisions for this page
        auto revRowsIt = rev_page_index.find(orig_page_id);
        if (revRowsIt == rev_page_index.end()) continue;

        for (int revRowNum : revRowsIt->second) {
            string revRow = getRowByIndex(revFile, revOffsetFile, revRowNum);
            vector<string> revCols = extractColumns(revRow, {0, 1, 2, 3, 4});

            int orig_rev_id = stoll(revCols[0]);
            string rev_minor = revCols[2];
            string rev_actor = revCols[3];
            string rev_time  = revCols[4];

            // Get text rows for this revision
            string old_text="";
            auto textRowsIt = old_id_index.find(orig_rev_id);
            if (textRowsIt != old_id_index.end() && !textRowsIt->second.empty()) {
                int textRowNum = textRowsIt->second.front(); // first (only) number
                string textRow = getRowByIndex(textFile, textOffsetFile, textRowNum);
                vector<string> textCols = extractColumns(textRow, {0, 1});
                old_text = textCols[1];
                
            }

            // Replicate revisions for each new page_id
            for (int i = 0; i < scale_factor; i++) {
                revFileNew << cur_old_id << "," << new_page_ids[i] << ","
                           << rev_minor << "," << rev_actor << "," << rev_time << "\n";

                textFileNew << cur_old_id << "," << old_text << "\n";
              
                cur_old_id++;
            }
        }
    }

    // Close files
    pageFile.close();
    pageExtraFile.close();
    pageFileNew.close();
    pageExtraFileNew.close();
    revFile.close();
    revFileNew.close();
    textFile.close();
    textFileNew.close();
    revOffsetFile.close();
    textOffsetFile.close();

    return 0;
}
