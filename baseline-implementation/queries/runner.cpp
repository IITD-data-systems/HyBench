#include "q1.cpp"
#include "q2.cpp"
#include "q3.cpp"
#include "q4.cpp"
#include "q5.cpp"
#include "q6.cpp"
#include "q7.cpp"
#include "q8.cpp"
#include "q9.cpp"
#include "q10.cpp"
#include "q11.cpp"
#include "q12.cpp"
#include "q13.cpp"
#include "q14.cpp"
#include "q15.cpp"
#include "q16.cpp"
#include "q17.cpp"
#include "q18.cpp"
#include "q19.cpp"
#include "q20.cpp"
#include "q21.cpp"
#include "q22.cpp"
#include "q23.cpp"
#include "q24.cpp"
#include "q25.cpp"
#include "q26.cpp"
#include "q27.cpp"
#include "q28.cpp"
#include "q29.cpp"
#include "q30.cpp"
#include "q31.cpp"
#include "q32.cpp"
#include "q33.cpp"
#include "q34.cpp"
#include "q35.cpp"
#include "q36.cpp"
#include "q37.cpp"
#include "q38.cpp"
#include "q39.cpp"
using namespace std;
int main(int argc, char* argv[]){
    int dim=384;
    string idx_kind=argv[1];
    string metric_type=argv[2];
    string queries_kind=argv[3];
    
    ifstream infile("../../database-generation/dim");
	infile >> dim;

    if(queries_kind=="text"){
    string indexFilename = "../../database-generation/index_files/text_" + idx_kind + "_" + metric_type + "_final.index";
    MyIndex* index;
    if(idx_kind=="faiss_ivfflat"){
        index = new IVFFlatIndex(indexFilename);
    }
    else if(idx_kind=="faiss_hnsw"){
        index = new HNSWIndex(indexFilename);
    }
    else if(idx_kind=="hnswlib"){
        
        index = new HNSWLibIndex(indexFilename,metric_type,dim);
    }
    index->get_stats();

    string rev_id_index_filename = "../../database-generation/index_files/rev_id_index.bin";
    std::unordered_map<int, std::vector<int>> rev_id_index;
    load_index(rev_id_index, rev_id_index_filename);

    cout<<"start_text"<<endl;


    // query 1
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        
        q1::q(query_size,index,search_parameter);
    }
    
    // query 2
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q2::q(query_size,index,search_parameter);
    }

    // query 5
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q5::q(query_size,index,rev_id_index,search_parameter);
    }

    // // query 6
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q6::q(query_size,index,rev_id_index,search_parameter);
    }

    // query 11
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q11::q(query_size,index,rev_id_index,search_parameter);
    }

    // // query 12
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q12::q(query_size,index,rev_id_index,search_parameter);
    }

    // query 16
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q16::q(query_size,index,search_parameter);
    }

    // query 17
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q17::q(query_size,index,search_parameter);
    }

    // query 20
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q20::q(query_size,index,rev_id_index,search_parameter);
    }

    // // query 21
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q21::q(query_size,index,rev_id_index,search_parameter);
    }


    // // query 26
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q26::q(query_size,index,search_parameter);
    }

    // query 27
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q27::q(query_size,index,search_parameter);
    }

    // // query 30
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q30::q(query_size,index,rev_id_index,search_parameter);
    }

    // // query 31
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q31::q(query_size,index,rev_id_index,search_parameter);
    }

    // // query 36
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q36::q(query_size,index,search_parameter);
    }

    // query 37
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q37::q(query_size,index,search_parameter);
    }

    // query 38
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q38::q(query_size,index,search_parameter);
    }

    // // query 39
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q39::q(query_size,index,search_parameter);
    }

    }
    else if (queries_kind=="page"){

    string indexFilename = "../../database-generation/index_files/page_" + idx_kind + "_" + metric_type + "_final.index";
    MyIndex* index;
    if(idx_kind=="faiss_ivfflat"){
        index = new IVFFlatIndex(indexFilename);
    }
    else if(idx_kind=="faiss_hnsw"){
        index = new HNSWIndex(indexFilename);
    }
    else if(idx_kind=="hnswlib"){
        
        index = new HNSWLibIndex(indexFilename,metric_type);
    }
    index->get_stats();

    string rev_page_index_filename = "../../database-generation/index_files/rev_page_index.bin";
    std::unordered_map<int, std::vector<int>> rev_page_index;
    load_index(rev_page_index, rev_page_index_filename);

    cout<<"start_page"<<endl;

    // query 3
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q3::q(query_size,index,search_parameter);
    }

    // // query 4
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q4::q(query_size,index,search_parameter);
    }
    // // query 7
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q7::q(query_size,index,rev_page_index,search_parameter);
    }

    // // query 8
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q8::q(query_size,index,rev_page_index,search_parameter);
    }
    
    // // query 9
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q9::q(query_size,index,rev_page_index,search_parameter);
    }

    // // query 10
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q10::q(query_size,index,rev_page_index,search_parameter);
    }

    // query 18
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q18::q(query_size,index,search_parameter);
    }

    // // query 19
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q19::q(query_size,index,search_parameter);
    }

    // query 22
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q22::q(query_size,index,rev_page_index,search_parameter);
    }

    // // query 23
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q23::q(query_size,index,rev_page_index,search_parameter);
    }

    // // query 28
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q28::q(query_size,index,search_parameter);
    }

    // // query 29
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q29::q(query_size,index,search_parameter);
    }

    // // query 32
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q32::q(query_size,index,rev_page_index,search_parameter);
    }

    // // query 33
    for(int _=0;_<30;_++){
        int search_parameter;
        string query_size;
        cin>>query_size;
        if(query_size=="done"){
            break;
        }
        cin>>search_parameter;
        q33::q(query_size,index,rev_page_index,search_parameter);
    }
    }
    else if(queries_kind=="te"){
        string old_id_index_filename = "../../database-generation/index_files/old_id_index.bin";
        unordered_map<int, vector<int>> old_id_index;
        load_index(old_id_index, old_id_index_filename);
        vector<vector<float>> embeddings=loadVectorOfVectors("../database-generation/text_embeddings.bin");
        
        cout<<"start"<<endl;

        // q13
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q13::q(query_size,old_id_index,embeddings,metric_type);
        } 

        // q14
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q14::q(query_size,old_id_index,embeddings,metric_type);
        } 

        // q24
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q24::q(query_size,old_id_index,embeddings,metric_type);
        } 

        // q25
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q25::q(query_size,old_id_index,embeddings,metric_type);
        } 

        // q34
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q34::q(query_size,old_id_index,embeddings,metric_type);
        } 

        // q35
        for(int _=0;_<30;_++){
            string query_size;
            cin>>query_size;
            if(query_size=="done"){
                break;
            }
            q35::q(query_size,old_id_index,embeddings,metric_type);
        } 
    }
    else{
        string page_id_index_filename = "../../database-generation/index_files/page_id_index.bin";
        string page_embedding_file =  "../../database-generation/data_csv_files/page_csv_files/embedding.csv";
        vector<vector<float>> embeddings=loadVectorOfVectors("../database-generation/page_embeddings.bin");
        unordered_map<int, vector<int>> page_id_index;
        load_index(page_id_index, page_id_index_filename);

        cout<<"start"<<endl;
        q15::q(page_id_index,embeddings,metric_type);
    }

    return 0;

    
}




