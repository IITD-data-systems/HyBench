import torch
from transformers import AutoTokenizer, AutoModel
import csv
import os
import sys

# Load MiniLM model and tokenizer
model='microsoft/MiniLM-L12-H384-uncased'
model=sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model)

# Set model to evaluation mode
model.eval()
left=True

page_title=["Quantum Computing","Artificial Intelligence","History of Mathematics"]
text_normal=["Climate change is one of the most pressing global issues of our time. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels like coal, oil, and gas. These activities release large amounts of greenhouse gases such as carbon dioxide into the atmosphere, trapping heat and causing the Earth’s temperature to rise. This warming leads to more frequent and severe weather events, including hurricanes, droughts, and floods. Rising temperatures also result in melting glaciers and polar ice, leading to sea level rise that threatens coastal communities worldwide. The effects of climate change are not evenly distributed; developing nations often face greater vulnerability due to limited resources and infrastructure. Efforts to combat climate change include reducing emissions through renewable energy, improving energy efficiency, and reforestation. The Paris Agreement, signed by most countries, aims to limit global warming to well below 2 degrees Celsius. While progress has been made, much more action is needed to avoid irreversible damage to ecosystems and human societies.","Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and make decisions. AI technologies have rapidly advanced in recent years, impacting industries ranging from healthcare to finance and transportation. At its core, AI involves machine learning—algorithms that allow computers to learn from data without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with many layers to model complex patterns. Applications of AI include image and speech recognition, natural language processing, robotics, and autonomous vehicles. AI systems are trained on large datasets to identify trends, make predictions, and automate tasks. However, AI also raises important ethical and societal questions, such as bias in algorithms, job displacement, and the need for transparency. Efforts are underway to develop fair, explainable, and trustworthy AI. Governments and organizations are drafting policies to ensure AI is developed and used responsibly. As the technology evolves, AI has the potential to both greatly benefit and disrupt society, making its careful governance essential for the future.","Quantum computing is a revolutionary approach to computation that harnesses the principles of quantum mechanics, such as superposition and entanglement, to perform operations on data. Unlike classical computers that use bits as the smallest unit of data, quantum computers use quantum bits or qubits. These qubits can represent both 0 and 1 simultaneously, allowing quantum machines to explore many possible solutions at once. This ability gives quantum computers exponential speed-ups for certain problems like factoring large numbers, searching unsorted databases, and simulating quantum systems. The development of quantum hardware, including superconducting circuits, ion traps, and topological qubits, is an active area of research. However, quantum systems are extremely sensitive to noise and decoherence, making error correction a major challenge. Despite these obstacles, major companies and research institutions are racing to achieve quantum advantage—the point at which a quantum computer can solve a problem that is practically impossible for any classical computer. As the field matures, quantum computing holds the promise of transforming cryptography, optimization, materials science, and more."]
text_related=["Urban transportation systems are essential to the movement of people and goods within cities. They include public transit networks like buses, trains, and subways, as well as private vehicles, bicycles, and pedestrian infrastructure. Efficient transportation is vital for economic productivity, reducing commute times, and ensuring accessibility. However, many urban areas struggle with traffic congestion, outdated infrastructure, and rising emissions. Urban planning increasingly emphasizes multi-modal integration and sustainable transportation to meet growing demand and environmental concerns.","Air pollution poses significant health and environmental risks worldwide. It originates from various sources, including industrial emissions, vehicular exhaust, and construction dust. In urban areas, traffic congestion and fossil fuel use are primary contributors. Prolonged exposure to polluted air can lead to respiratory diseases, cardiovascular problems, and reduced life expectancy. Fine particulate matter (PM2.5) and nitrogen dioxide are among the most harmful pollutants. Governments and organizations are working to reduce air pollution through stricter regulations, cleaner fuels, and public awareness campaigns.","Smart cities use digital technologies to enhance urban life through data-driven governance, improved infrastructure, and citizen engagement. IoT sensors, cloud platforms, and real-time data collection allow for smarter traffic management, efficient energy use, and better waste handling. These cities aim to improve livability, sustainability, and resilience to challenges like climate change and rapid urbanization. Key innovations include adaptive street lighting, predictive maintenance for public services, and platforms for community feedback."]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    full_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    if left:
        with open('dim', 'w') as f:
            f.write(str(len(full_embedding)) + '\n')
        left=False    
    return full_embedding.tolist()  

# Main processing function
import numpy as np

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def page_embeddings_generation():
    input_path = "data_csv_files/page_csv_files/page.csv"
    output_path = "data_csv_files/page_csv_files/embedding.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    l2_distances = []
    cos_distances = []
    query_embedding = A_page[0]

    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            text = row[1]
            embedding = get_embedding(text)
            writer.writerow([str(embedding)])


            emb_np = np.array(embedding)

            if query_embedding is None:
                query_embedding = emb_np
            else:
                l2 = l2_distance(emb_np, query_embedding)
                cos = cosine_distance(emb_np, query_embedding)
                l2_distances.append(l2)
                cos_distances.append(cos)

            if i % 1000 == 0:
                print(f"Processed {i} rows...")

    # Get 50th, 100th, ..., 500th nearest neighbors
    step_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    l2_distances.sort()
    cos_distances.sort()

    page_distances_l2 = [(k, l2_distances[k - 1]) for k in step_values if k <= len(l2_distances)]
    page_distances_cos = [(k, cos_distances[k - 1]) for k in step_values if k <= len(cos_distances)]

    print("Page Embeddings generation complete.")
    return page_distances_l2,page_distances_cos


def text_embeddings_generation():
    input_path = "data_csv_files/text_csv_files/text.csv"
    output_path = "data_csv_files/text_csv_files/embedding.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    l2_distances = []
    cos_distances = []
    l2_distances1 = []
    cos_distances1 = []
    query_embedding = A_text[0]
    query_embedding1 = A_related[1]

    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            text = row[1]
            embedding = get_embedding(text)
            writer.writerow([str(embedding)])


            emb_np = np.array(embedding)

            if query_embedding is None:
                query_embedding = emb_np
            else:
                l2 = l2_distance(emb_np, query_embedding)
                cos = cosine_distance(emb_np, query_embedding)
                l2_distances.append(l2)
                cos_distances.append(cos)
                l2 = l2_distance(emb_np, query_embedding1)
                cos = cosine_distance(emb_np, query_embedding1)
                l2_distances1.append(l2)
                cos_distances1.append(cos)

            if i % 1000 == 0:
                print(f"Processed {i} rows...")

    # Get 50th, 100th, ..., 500th nearest neighbors
    step_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    l2_distances.sort()
    cos_distances.sort()
    l2_distances1.sort()
    cos_distances1.sort()
    
    text_distances_l2 = [(k, l2_distances[k - 1]) for k in step_values if k <= len(l2_distances)]
    text_distances_cos = [(k, cos_distances[k - 1]) for k in step_values if k <= len(cos_distances)]
    related_distances_l2 = [(k, l2_distances1[k - 1]) for k in step_values if k <= len(l2_distances1)]
    related_distances_cos = [(k, cos_distances1[k - 1]) for k in step_values if k <= len(cos_distances1)]
    
    print("text Embeddings generation complete.")
    return text_distances_l2,text_distances_cos,related_distances_l2,related_distances_cos

# generate_query.py
import sys



################################################################################################################################################################################################    
# QUERY 1 
def q1():
    qn=1
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
    



################################################################################################################################################################################################    
# QUERY 2
def q2():
    qn=2
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(text_dist[op][i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 3
def q3():
    qn=3
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(k_limits[i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 4
def q4():
    qn=4
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(page_dist[op][i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 5
def q5():
    qn=5
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(k_limits[i],DATELOW[i],DATEHIGH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 6
def q6():
    qn=6
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(text_dist[op][i],DATELOW[i],DATEHIGH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 7
def q7():
    qn=7
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
     
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(k_limits[i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 8
def q8():
    qn=8
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(page_dist[op][i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 9
def q9():
    qn=9
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 10
def q10():
    qn=10
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(page_dist[op][i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 11
def q11():
    qn=11
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 12
def q12():
    qn=12
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(text_dist[op][i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 13
def q13():
    qn=13
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(k_limits[i],YEARL[i],YEARH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
    
    return

    
################################################################################################################################################################################################    
# QUERY 14
def q14():
    qn=14
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(text_dist[op][i],YEARL[i],YEARH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
                 
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 15
def q15():
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(15,15,opn)
    return

    
################################################################################################################################################################################################    
# QUERY 16
def q16():
    qn=16
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(offsets[i]+1,offsets[i]+k_limits[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
                
              
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 17
def q17():
    qn=17
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(text_dist_ranges[op][i][0],text_dist_ranges[op][i][1]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
                
            
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    



################################################################################################################################################################################################    
# QUERY 18
def q18():
    qn=18
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(offsets[i]+1,offsets[i]+k_limits[i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
                
            
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")

    
################################################################################################################################################################################################    
# QUERY 19
def q19():
    qn=19
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)

    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(page_len_constraints[i],page_dist_ranges[op][i][0],page_dist_ranges[op][i][1]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
                
            
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")


################################################################################################################################################################################################    
# QUERY 20
def q20():
    qn=20
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(offsets[i]+1,offsets[i]+k_limits[i],DATELOW[i],DATEHIGH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")


################################################################################################################################################################################################    
# QUERY 21
def q21():
    qn=21
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(text_dist_ranges[op][i][0],text_dist_ranges[op][i][1],DATELOW[i],DATEHIGH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
    
################################################################################################################################################################################################    
# QUERY 22
def q22():
    qn=22
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(offsets[i]+1,offsets[i]+k_limits[i],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
        
################################################################################################################################################################################################    
# QUERY 23
def q23():
    
    qn=23
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(page_dist_ranges[op][i][0],page_dist_ranges[op][i][1],page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
################################################################################################################################################################################################    
# QUERY 24
def q24():
    qn=24
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(offsets[i]+1,offsets[i]+k_limits[i],YEARL[i],YEARH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
      
      
################################################################################################################################################################################################    
# QUERY 25
def q25():
    qn=25
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(text_dist_ranges[op][i][0],text_dist_ranges[op][i][1],YEARL[i],YEARH[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 26
def q26():
    qn=26
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(len(ranks_list[i])," ".join(map(str, ranks_list[i]))))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
        
################################################################################################################################################################################################    
# QUERY 27
def q27():
    qn=27
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(len(text_distance_ranges_list[op][i])))
            for j in text_distance_ranges_list[op][i]:
                f.write("{} {} ".format(*j))
            
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 28
def q28():
    qn=28
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(len(ranks_list[i])," ".join(map(str, ranks_list[i])),page_len_constraints[i]))
            
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 29
def q29():
    qn=29
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(len(page_distance_ranges_list[op][i])))
            
            for j in page_distance_ranges_list[op][i]:
                f.write("{} {} ".format(*j))
            f.write("{} ".format(page_len_constraints[i]))    
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 30
def q30():
    qn=30
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(len(ranks_list[i])," ".join(map(str, ranks_list[i])),DATELOW[i],DATEHIGH[i]))
           
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 31
def q31():
    qn=31
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(len(text_distance_ranges_list[op][i])))
            for j in text_distance_ranges_list[op][i]:
                f.write("{} {} ".format(*j))
            f.write("{} {} ".format(DATELOW[i],DATEHIGH[i]))    
           
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 32
def q32():
    qn=32
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} ".format(len(ranks_list[i])," ".join(map(str, ranks_list[i])),page_len_constraints[i]))
           
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 33
def q33():
    qn=33
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(len(page_distance_ranges_list[op][i])))
            for j in page_distance_ranges_list[op][i]:
                f.write("{} {} ".format(*j))
            f.write("{} ".format(page_len_constraints[i]))    
           
            # Then write floats from 1.0 to 384.0
            for j in A_page[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 34
def q34():
    qn=34
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} {} {} ".format(len(ranks_list[i])," ".join(map(str, ranks_list[i])),YEARL[i],YEARH[i]))
           
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 35
def q35():
    qn=35
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(len(text_distance_ranges_list[op][i])))
            for j in text_distance_ranges_list[op][i]:
                f.write("{} {} ".format(*j))
            f.write("{} {} ".format(YEARL[i],YEARH[i]))    
           
            # Then write floats from 1.0 to 384.0
            for j in A_text[0]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 36
def q36():
    qn=36
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))  
           
            # Then write floats from 1.0 to 384.0
            for j in A_related[0]:
                f.write(f" {float(j)}")
                
            for j in A_related[1]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 37
def q37():
    qn=37
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))  
           
            # Then write floats from 1.0 to 384.0
            for j in A_related[0]:
                f.write(f" {float(j)}")
                
            for j in A_related[1]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 38
def q38():
    qn=38
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} {} ".format(k_limits[i],related_dist[op][i]))  
           
            # Then write floats from 1.0 to 384.0
            for j in A_related[0]:
                f.write(f" {float(j)}")
                
            for j in A_related[1]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
        
################################################################################################################################################################################################    
# QUERY 39
def q39():
    qn=39
    output_filename = "../query-generation/q{}_queries/q{}_queries_{}_1.txt".format(qn,qn,opn)
    
    for i in range(3):
        
        output_filename=output_filename[:-5]+str(i+1)+output_filename[-4:]
        
        
        with open(output_filename, "w") as f:
            # Write 40 first
            f.write("{} ".format(k_limits[i]))  
           
            # Then write floats from 1.0 to 384.0
            for j in A_related[0]:
                f.write(f" {float(j)}")
                
            for j in A_related[1]:
                f.write(f" {float(j)}")
            
                
            f.write("\n")  # optional newline at the end

        print(f"Query written to {output_filename}")
################################################################################################################################################################################################    
A_page=[[-0.2984125316143036, 0.3702804744243622, 0.09224986284971237, 0.10958653688430786, 0.06719369441270828, -0.24571451544761658, 0.15149323642253876, 0.0655442401766777, -0.12504373490810394, -0.0005787322297692299, 0.05305050313472748, -0.27781227231025696, 0.040995821356773376, 0.20484715700149536, 0.1970844864845276, 0.0681488886475563, -0.07218935340642929, 0.005856251809746027, -0.07557722926139832, -0.0800028070807457, -0.16117334365844727, -0.1005835011601448, -0.059764616191387177, -0.2016800045967102, 0.1341465711593628, 0.017616719007492065, -0.2146591693162918, -0.14453500509262085, 0.45480695366859436, -1.7796369791030884, -0.11651618033647537, -0.1277334839105606, -0.03605896607041359, -0.11517353355884552, -0.1636555790901184, -0.05668223276734352, -0.07258313149213791, 0.03521796688437462, -0.14153505861759186, 0.012535725720226765, -0.1019166111946106, -0.0015023441519588232, -0.11449473351240158, -0.13019713759422302, 0.0020954497158527374, 0.10563336312770844, 0.20280231535434723, 0.0029207663610577583, -0.1466372311115265, -0.06512509286403656, -0.02626996859908104, 0.08376099914312363, 0.037520118057727814, -0.004116256721317768, 0.3076000213623047, -0.05174581706523895, 0.21755261719226837, -0.08564522862434387, 0.03067876026034355, 0.006025597453117371, 0.18128949403762817, 0.06815695762634277, -1.5887926816940308, 0.03351067751646042, 0.3384034335613251, -0.03406316041946411, 0.02381480485200882, 0.001168171875178814, -0.03581523522734642, 0.31914272904396057, -0.060105595737695694, 0.11587078124284744, 0.0369875393807888, 0.18158717453479767, 0.18357497453689575, 0.14732448756694794, 0.06543760746717453, 0.048238687217235565, 0.08446507900953293, 0.08482133597135544, -0.06091521307826042, -0.07603371143341064, -0.20098115503787994, 0.044351983815431595, 0.05865536257624626, 0.10819240659475327, -0.17015089094638824, -0.10500364750623703, 0.1827024668455124, -0.038668401539325714, -0.08603371679782867, 0.10961561650037766, -0.022194169461727142, -0.1176614835858345, -0.6547435522079468, 0.06908295303583145, 0.051047977060079575, -0.21783341467380524, 0.009821247309446335, 3.967729091644287, 0.054519202560186386, -0.10327649116516113, 0.0598001554608345, -0.006000662688165903, -0.10425581783056259, -0.06242837756872177, -0.09916631877422333, 0.03700894117355347, 0.009982015937566757, -0.10561506450176239, 0.08810052275657654, -0.052812062203884125, 0.03470636159181595, -0.050589028745889664, 0.0213834997266531, 0.17585822939872742, 0.1048455536365509, -0.02103927731513977, -0.04982289671897888, 0.22447660565376282, -0.02793074958026409, -0.018254932016134262, -0.09324156492948532, 0.22151483595371246, -0.09132478386163712, -1.290431261062622, 0.12529729306697845, -0.0333126075565815, -0.006634388118982315, 0.05853114277124405, 0.09965935349464417, -0.3109102249145508, 0.22317150235176086, 0.18789887428283691, -0.033682361245155334, 0.03348687291145325, -0.10567818582057953, -0.10790777206420898, 0.17172013223171234, -0.25013235211372375, -0.08231114596128464, 0.031116709113121033, -0.18476754426956177, -0.2722242474555969, -0.06589358299970627, 0.6773286461830139, -0.029174499213695526, -0.028061795979738235, -0.07479050010442734, 0.2006150633096695, 0.01646910049021244, 0.20716431736946106, 0.042980559170246124, 0.09969916194677353, 0.1113404631614685, -0.08917150646448135, -0.010836424306035042, 0.8380598425865173, -0.07723984122276306, -0.01941656693816185, -0.11403102427721024, -0.24418914318084717, -0.07839131355285645, 0.6023597121238708, 0.03044312447309494, -0.3212355077266693, -0.0813632532954216, 0.06531155109405518, -0.000677129253745079, 0.34156185388565063, 0.05634221062064171, -0.08451864868402481, 0.12837904691696167, 0.009886564686894417, 0.31852638721466064, 0.13676469027996063, -0.022030403837561607, 0.06266970187425613, -0.021990720182657242, 0.02763378620147705, 0.011833537369966507, -0.04072397202253342, 0.29968926310539246, 0.18249426782131195, 0.1505826860666275, 0.1468948870897293, 0.01217967364937067, 0.009856689721345901, 0.0027726497501134872, 0.22518020868301392, -0.19560419023036957, 0.10223881900310516, -0.04550274461507797, 0.15127289295196533, -0.09047617763280869, -0.17151838541030884, 0.08886826783418655, 0.004810880403965712, -0.28362125158309937, -0.13485316932201385, 0.25035709142684937, -0.10812608152627945, 0.22606897354125977, -0.06908358633518219, -0.22262075543403625, -0.20400488376617432, 0.24455223977565765, -0.032946888357400894, -0.24047347903251648, -0.1820565164089203, 0.03374225273728371, -0.15152058005332947, 0.07203212380409241, -0.019991066306829453, 0.028217677026987076, -1.068180426955223e-05, -0.2702453136444092, -0.053209565579891205, -0.09301553666591644, -0.019196372479200363, 0.2518855929374695, -0.1149105355143547, -0.27466127276420593, -2.446105480194092, -0.4300864636898041, -0.2536069452762604, -0.0503687746822834, 0.36621201038360596, 0.030609529465436935, 0.058103419840335846, 0.13015013933181763, -0.006308360956609249, -0.22031697630882263, 0.6010809540748596, 0.3024105131626129, -0.046483010053634644, -0.23039884865283966, 0.2583978474140167, 0.15163542330265045, -0.2036702036857605, 0.12781935930252075, 0.03916104882955551, -0.008250528946518898, 0.0072693657130002975, -0.11461997777223587, -0.10341131687164307, -0.15653228759765625, 0.06528133153915405, 0.0043115634471178055, 1.7981793880462646, 0.019271377474069595, 0.18481124937534332, 0.34084364771842957, 0.1125115230679512, -0.07051581144332886, 0.08537886291742325, -0.19521701335906982, -3.74307855963707e-05, -0.002932053990662098, -0.056110039353370667, 0.22224879264831543, 0.008114764466881752, -0.015264667570590973, -0.10295555740594864, -0.031630925834178925, 0.08889725059270859, 0.18528582155704498, 0.027393702417612076, -0.2262832522392273, -0.06723709404468536, -0.21239574253559113, 0.18800465762615204, -0.24770738184452057, -0.0029430091381073, -0.119363933801651, 0.10215446352958679, 0.09027810394763947, -0.0656399130821228, -0.022599894553422928, -0.2212325483560562, -0.2419356107711792, 0.06862711906433105, -0.21458378434181213, -0.004884649999439716, 0.046074412763118744, -0.06416597962379456, 0.0580875426530838, -0.20829439163208008, -0.2584109902381897, -0.00933015439659357, -0.09707198292016983, 0.2832646369934082, -0.14583075046539307, -0.12769024074077606, 0.44485050439834595, 0.11289986968040466, 0.13056138157844543, 0.20515857636928558, -0.011002538725733757, -0.17876005172729492, 0.24872329831123352, -0.07304541766643524, -0.10118355602025986, 0.2773122191429138, 0.2694177031517029, 0.03301198035478592, -0.002316005527973175, 0.13903190195560455, 0.3551487922668457, -0.09923990815877914, 0.17191994190216064, -0.16130021214485168, 0.133289635181427, -0.06595371663570404, -0.10230839252471924, 0.2900697588920593, -0.18138852715492249, 0.2884555757045746, 0.050880953669548035, -2.5982296466827393, 0.14283958077430725, -0.07564447820186615, -0.0035387612879276276, -0.045414961874485016, 0.023129602894186974, 0.20008832216262817, 0.14971622824668884, -0.07629840075969696, 0.026247575879096985, -0.1715177446603775, 0.2765951156616211, 0.08054053783416748, -0.14223481714725494, -0.059325478971004486, 0.2168666124343872, 0.10557369142770767, 0.010834931395947933, -0.21430069208145142, -0.7719459533691406, -0.18221880495548248, -0.16499100625514984, 1.1639738082885742, -0.11873162537813187, 0.11320845037698746, -0.1303885132074356, 0.08306194841861725, -0.03391212597489357, -0.2357056736946106, 0.051824651658535004, 0.05600233003497124, -0.0845811665058136, 0.17416062951087952, 0.062304750084877014, 0.1221553310751915, -0.17369209229946136, -0.04762393236160278, 0.07302805781364441, 0.08339206874370575, 0.06294453144073486, 0.04898209124803543, 0.04340086132287979, -0.002696068026125431, 0.08534148335456848, 0.3532260060310364, -0.13522306084632874, -0.09623131155967712, -0.11924152076244354, -0.00447462685406208, -0.014854508452117443, 0.09803401678800583, -0.11135156452655792, 0.3028623163700104, 0.05190085247159004, 0.03267643600702286, 0.09914544969797134, -0.1315092146396637, 0.0023972662165760994, -0.08470448106527328, -0.09594038128852844, 0.06100340187549591, 0.13254353404045105, -0.1765356808900833, -0.054679788649082184, -0.3712432384490967],[-0.3096412718296051, 0.36006343364715576, 0.0578293576836586, 0.05964819714426994, 0.11559107899665833, -0.21513418853282928, 0.2568858563899994, -0.02501235157251358, -0.03403172269463539, 0.1023576632142067, 0.13654792308807373, -0.30212435126304626, 0.09991960227489471, 0.12493419647216797, 0.24199798703193665, 0.043245844542980194, -0.0319146029651165, -0.11687098443508148, -0.40512514114379883, -0.187143474817276, -0.03011905401945114, -0.2157098948955536, 0.06322845816612244, -0.17699483036994934, 0.040214091539382935, 0.16316261887550354, -0.1888783872127533, -0.15150123834609985, 0.20404398441314697, -1.9375145435333252, -0.04572494700551033, -0.12584830820560455, 0.36565476655960083, -0.04283307120203972, -0.20343440771102905, -0.04331938922405243, -0.11910746991634369, -0.0015284931287169456, -0.13212227821350098, 0.02269422821700573, -0.15071740746498108, 0.030291080474853516, -0.150913804769516, -0.1474168598651886, -0.029310403391718864, 0.1175980418920517, 0.19640524685382843, 0.07738406956195831, 0.024759382009506226, -0.031269416213035583, 0.0715511366724968, 0.11292138695716858, -0.028596000745892525, 0.06536923348903656, 0.2480117380619049, 0.08652719110250473, 0.1609480232000351, -0.08686285465955734, 0.02379443123936653, 0.10093814134597778, 0.21575535833835602, 0.0020293162669986486, -1.5429538488388062, 0.12404517084360123, 0.1908615529537201, -0.01661153882741928, 0.000657401978969574, -0.14445531368255615, 0.023798618465662003, 0.05977277457714081, -0.0993758887052536, 0.0890469178557396, -0.06500183790922165, 0.003683166578412056, 0.22681821882724762, 0.1422201693058014, -0.014772433787584305, -0.07436169683933258, 0.17304809391498566, 0.051850318908691406, -0.002828954253345728, -0.11772987246513367, -0.08866176009178162, 0.1462917923927307, 0.11348768323659897, 0.049271680414676666, -0.18701836466789246, -0.007444074377417564, 0.09552116692066193, -0.01827523484826088, -0.11330022662878036, 0.05285787582397461, 0.0772651806473732, -0.07779931277036667, -0.582548201084137, 0.024837082251906395, 0.04463307932019234, -0.39216190576553345, -0.26780495047569275, 3.8482367992401123, -0.022363847121596336, -0.06962325423955917, -0.10274741053581238, 0.034347619861364365, 0.0010714121162891388, -0.15888851881027222, -0.06799418479204178, -0.10792713612318039, 0.04814079403877258, -0.013389627449214458, 0.021480105817317963, -0.13368481397628784, -0.04787392169237137, -0.08812227100133896, 0.16581453382968903, 0.19954220950603485, 0.14222252368927002, -0.02608834020793438, -0.0369199737906456, 0.1404358595609665, 0.03311668336391449, 0.003540322184562683, -0.08369164168834686, 0.22347195446491241, 0.08118737488985062, -1.3948109149932861, 0.12955057621002197, 0.025424379855394363, -0.1069720983505249, 0.02981562353670597, 0.04564857482910156, -0.20854559540748596, 0.030029691755771637, 0.1792168766260147, -0.013059297576546669, 0.03833168372511864, -0.10961958765983582, -0.2048460990190506, 0.13882681727409363, -0.09910023957490921, -0.20445813238620758, 0.003626415506005287, -0.00915982760488987, -0.15076987445354462, -0.2024296075105667, 0.9215869903564453, 0.14199961721897125, -0.03988094627857208, -0.016428550705313683, 0.27269530296325684, 0.005978046916425228, 0.22308729588985443, 0.10162942856550217, 0.07775245606899261, 0.05681731551885605, -0.0049610137939453125, 0.19795489311218262, 0.6591836810112, -0.142075315117836, -0.03724547103047371, -0.06443280726671219, -0.2510688602924347, -0.08532906323671341, 0.5050445199012756, 0.0783577486872673, -0.37378525733947754, -0.052700240164995193, 0.05567631497979164, 0.006980583071708679, 0.3810085654258728, 0.06339766085147858, -0.07151154428720474, 0.1509629487991333, -0.05288407951593399, 0.3307192623615265, 0.0712580606341362, 0.1763206273317337, 0.16980762779712677, -0.041117846965789795, -0.014418743550777435, 0.05556673929095268, -0.019062399864196777, 0.181584894657135, 0.13244451582431793, 0.10177507996559143, 0.270383358001709, 0.1172892302274704, 0.006184871774166822, -0.01830362342298031, 0.37364381551742554, -0.20385703444480896, 0.052293308079242706, -0.0492582842707634, 0.057380311191082, -0.08896913379430771, -0.1601153165102005, 0.043049439787864685, 0.011329550296068192, -0.21856172382831573, -0.22590813040733337, 0.04680148512125015, -0.028187889605760574, 0.16556015610694885, -0.03720872104167938, -0.2535553574562073, -0.17244061827659607, 0.11089769750833511, 0.054894059896469116, -0.28499260544776917, -0.21355609595775604, 0.07077419012784958, -0.11580255627632141, 0.11084482818841934, -0.02051270380616188, 0.02944040857255459, 0.04634040966629982, -0.21955347061157227, -0.04228862747550011, 0.01639011688530445, -0.01365357730537653, 0.14421330392360687, -0.14280328154563904, -0.1117091178894043, -2.4276931285858154, -0.36036235094070435, -0.3570345640182495, -0.1838424801826477, 0.41555076837539673, 0.00117426086217165, 0.10206088423728943, 0.11081403493881226, 0.07036174088716507, -0.03573768585920334, 0.6848428845405579, 0.14353714883327484, 0.003752943128347397, -0.1246771439909935, 0.2332964539527893, 0.27573537826538086, -0.24038180708885193, 0.008225403726100922, 0.1004265621304512, -0.017154177650809288, 0.0460977703332901, -0.20799700915813446, 0.20048834383487701, -0.14635775983333588, 0.0933903306722641, -0.0254556592553854, 1.831468939781189, -0.16720277070999146, 0.35172176361083984, 0.2959998548030853, 0.1328168511390686, -0.10569511353969574, 0.07812775671482086, -0.23018863797187805, 0.05339616909623146, 0.05263929069042206, 0.00953984446823597, 0.15759767591953278, -0.13559909164905548, 0.00794706679880619, -0.01816963590681553, 0.05433640629053116, 0.07913453876972198, 0.021876689046621323, 0.11392524838447571, -0.18052229285240173, -0.0797213688492775, -0.24846309423446655, 0.03812859579920769, -0.2017897665500641, 0.023540541529655457, -0.07909248024225235, 0.10129080712795258, 0.07061846554279327, -0.15133258700370789, -0.02600611373782158, -0.2406427264213562, -0.20391151309013367, 0.006122433580458164, -0.0708552896976471, 0.0005448032170534134, 0.06496020406484604, -0.04148675128817558, -0.0009763911366462708, -0.216353639960289, -0.24123705923557281, -0.024260791018605232, -0.0682445839047432, 0.31760674715042114, -0.08807142823934555, -0.19753512740135193, 0.5787035822868347, 0.009355934336781502, 0.016209227964282036, 0.2583937346935272, -0.1213623434305191, -0.04378748685121536, 0.2403409779071808, -0.11914636939764023, -0.10317809879779816, 0.2426527738571167, 0.23707590997219086, -0.007629899308085442, 0.033763591200113297, 0.15475572645664215, 0.28766682744026184, 0.0005458444356918335, 0.08916817605495453, -0.28647133708000183, 0.11720655858516693, 0.0315723791718483, -0.050931621342897415, 0.22033728659152985, -0.12257809937000275, 0.4431399405002594, 0.08268707245588303, -2.4973039627075195, 0.14471517503261566, -0.08204355835914612, 0.2538697123527527, 0.010532870888710022, -0.03933718800544739, 0.2124510109424591, 0.034394245594739914, -0.06573710590600967, 0.08365877717733383, -0.25389760732650757, 0.23646840453147888, 0.03828410431742668, -0.08036678284406662, -0.024989886209368706, 0.26488691568374634, 0.09233860671520233, -0.03754868730902672, -0.049434684216976166, -0.8736771941184998, -0.12380173802375793, -0.12659282982349396, 1.1010946035385132, -0.22086168825626373, -0.09680598974227905, -0.09593424946069717, 0.07261034846305847, -0.005456934217363596, -0.19451919198036194, -0.010772036388516426, 0.0012391344644129276, -0.10573209077119827, 0.12385968863964081, 0.138650044798851, 0.12506242096424103, -0.0327330119907856, -0.030076976865530014, 0.0061402879655361176, 0.017626916989684105, -0.005801232997328043, -0.27596116065979004, 0.0431848019361496, -0.1931101679801941, 0.09686632454395294, 0.3417659103870392, -0.18806232511997223, -0.07257314026355743, 0.007050216197967529, -0.04415328800678253, 0.06564153730869293, 0.0336403027176857, -0.18659475445747375, 0.3450869619846344, 0.022120436653494835, 0.1613813191652298, 0.055532995611429214, -0.1310230940580368, 0.07048357278108597, -0.060656800866127014, -0.13432638347148895, 0.10309296101331711, 0.10435984283685684, -0.015075134113430977, -0.10626368969678879, -0.4059979021549225],[-0.2566860318183899, 0.305233895778656, 0.023762691766023636, 0.020287977531552315, -0.07738396525382996, -0.1780623197555542, 0.22025060653686523, 0.14125826954841614, -0.016690082848072052, -0.12164285033941269, -0.005480816587805748, -0.05303674936294556, -0.0013301052385941148, 0.056832455098629, 0.04921508580446243, 0.04173871874809265, -0.013003523461520672, 0.11679930984973907, -0.35324016213417053, -0.07853575795888901, -0.36346495151519775, -0.014202581718564034, 0.011660447344183922, -0.04816689342260361, 0.1596836894750595, 0.06380769610404968, -0.10210512578487396, -0.0384628102183342, 0.37637859582901, -1.6061718463897705, -0.04844774305820465, -0.13021406531333923, 0.23075351119041443, -0.006557573564350605, -0.12134013324975967, -0.01107487641274929, -0.03230481594800949, 0.030415501445531845, -0.025093013420701027, 0.09372705221176147, -0.048186369240283966, 0.20112788677215576, -0.16740871965885162, -0.10936963558197021, 0.0508929006755352, 0.04263842850923538, 0.08560032397508621, -0.04416996240615845, -0.13404498994350433, -0.014433617703616619, -0.09715650230646133, 0.06589845567941666, -0.12400700151920319, -0.061666034162044525, 0.04056500643491745, 0.0037998207844793797, 0.15819406509399414, -0.02624550461769104, -0.01476369984447956, 0.05431298166513443, 0.10292283445596695, 0.0711074247956276, -1.616066336631775, 0.31106168031692505, 0.00022122636437416077, -0.011098935268819332, -0.021770652383565903, 0.21074433624744415, -0.033959634602069855, 0.12919309735298157, -0.032728541642427444, 0.08813810348510742, -0.07006913423538208, 0.18302085995674133, 0.16286709904670715, 0.08184132725000381, 0.06374572217464447, -0.038644835352897644, -0.09171660244464874, -0.019765418022871017, 0.12404515594244003, -0.07049703598022461, -0.1246456652879715, 0.0030382289551198483, 0.03220716863870621, 0.008046780712902546, -0.10792515426874161, -0.05682043358683586, 0.021813256666064262, -0.05107805132865906, 0.06726516038179398, -0.10513553768396378, 0.07453931123018265, -0.09516999870538712, -0.37578755617141724, 0.3316374123096466, 0.05110190436244011, -0.16256225109100342, 0.1802515685558319, 3.175351858139038, -0.05528554320335388, 0.009325506165623665, 0.1298406720161438, 0.09350533783435822, -0.0642479807138443, -0.050198398530483246, -0.07448705285787582, -0.10815338790416718, -0.061229754239320755, -0.16698068380355835, 0.11037905514240265, -0.03955135494470596, -0.09320640563964844, -0.08486716449260712, 0.14848747849464417, -0.010363774374127388, 0.27139222621917725, -0.010802064090967178, 0.11161845922470093, 0.18572261929512024, 0.07849808037281036, -0.03472517058253288, 0.004622875712811947, 0.12322697788476944, -0.1734754592180252, -1.1033995151519775, 0.15001626312732697, 0.2910553514957428, -0.061156224459409714, -0.057829149067401886, 0.18088963627815247, -0.29746395349502563, 0.10136513411998749, 0.1448678970336914, -0.09504888951778412, -0.0020244226325303316, 0.08214826881885529, -0.2478702813386917, 0.20537082850933075, -0.03439731523394585, -0.06003423407673836, -0.24112394452095032, -0.11583848297595978, -0.39232322573661804, -0.08830370008945465, 0.9077752828598022, -0.06713487207889557, -0.04414757341146469, -0.0966501459479332, 0.35480251908302307, 0.09920576959848404, 0.14860542118549347, -0.030634945258498192, 0.16248400509357452, -0.012398950755596161, -0.08387230336666107, 0.03181113675236702, 0.5621813535690308, -0.09313483536243439, 0.012845441699028015, -0.08646130561828613, -0.13283386826515198, -0.04379934072494507, 0.5942150354385376, 0.09863617271184921, -0.3835117816925049, -0.04018091410398483, 0.010983580723404884, 0.022053232416510582, 0.20017921924591064, 0.008712967857718468, -0.028432905673980713, 0.027164380997419357, 0.16434022784233093, 0.2584686279296875, 0.10158083587884903, 0.14762984216213226, 0.025825459510087967, -0.09916284680366516, -0.03133188933134079, -0.008098684251308441, 0.032336264848709106, 0.17482087016105652, 0.12675637006759644, 0.050803083926439285, 0.14275965094566345, -0.018066486343741417, 0.1669866293668747, -0.023872727528214455, 0.26001614332199097, 0.030791956931352615, 0.20023858547210693, -0.13773858547210693, -0.06850431859493256, -0.09097985178232193, -0.06477968394756317, 0.09500226378440857, -0.06504131853580475, -0.24128198623657227, -0.057653021067380905, -0.05237675458192825, 0.14231519401073456, 0.2973009943962097, 0.0545099675655365, -0.1471463292837143, -0.2536233067512512, -0.026592496782541275, -0.027359435334801674, -0.24925272166728973, -0.03037838265299797, -0.044058579951524734, -0.26134586334228516, 0.11661676317453384, -0.007313667330890894, 0.0014411464799195528, -0.06289876997470856, -0.05384643003344536, 0.008624526672065258, -0.04975033923983574, -0.01564941741526127, 0.010018279775977135, -0.11641526222229004, -0.08622260391712189, -2.566857099533081, -0.30294445157051086, -0.12038618326187134, -0.07532171905040741, 0.1671983301639557, -0.030817624181509018, 0.057495761662721634, -0.008835716173052788, 0.08913572877645493, -0.027148038148880005, 0.46987348794937134, 0.18477238714694977, -0.10303662717342377, 0.06587857753038406, 0.14619944989681244, 0.08355073630809784, -0.3393777310848236, 0.13180553913116455, -0.06679905951023102, -0.09039346873760223, 0.026791399344801903, -0.07510125637054443, 0.192936509847641, -0.42799320816993713, -0.13825321197509766, 0.0666460320353508, 1.73331618309021, -0.20591039955615997, 0.11761458963155746, 0.1431310474872589, 0.16287483274936676, -0.07864206284284592, 0.03549081087112427, -0.15594741702079773, 0.038831211626529694, -0.11538150161504745, -0.03934088349342346, 0.22163435816764832, -0.09038171917200089, 0.12436119467020035, -0.09987343847751617, 0.0069975825026631355, 0.11309964954853058, 0.2618461847305298, -0.07765638828277588, -0.13312415778636932, -0.10472802072763443, -0.15865084528923035, 0.04046960547566414, -0.01235278882086277, 0.04825516417622566, 0.011151199229061604, 0.03839057683944702, 0.20664827525615692, -0.19669660925865173, 0.09688442200422287, -0.4122390151023865, -0.21480801701545715, 0.06069456413388252, -0.10782400518655777, 0.01021132804453373, -0.07507918030023575, 0.053432513028383255, 0.062029432505369186, -0.11872515827417374, -0.2131618708372116, -0.047550447285175323, -0.008928073570132256, 0.39451584219932556, -0.14036825299263, -0.23032107949256897, 0.3317665457725525, 0.1309187412261963, -0.3040798306465149, 0.4254186153411865, -0.09690321981906891, -0.0004618443490471691, 0.01303921453654766, -0.025427568703889847, -0.0898725837469101, 0.19469523429870605, 0.1181638091802597, 0.0407378226518631, 0.06727669388055801, 0.11297498643398285, 0.23885324597358704, -0.07730613648891449, 0.025632355362176895, -0.08078885823488235, 0.026275401934981346, -0.047160688787698746, 0.08699101954698563, 0.13022716343402863, -0.16738085448741913, 0.32468438148498535, 0.11501016467809677, -2.8838438987731934, 0.10719047486782074, -0.0862848311662674, 0.08969849348068237, 0.010631215758621693, -0.07815241813659668, 0.09924714267253876, 0.3621537387371063, 0.005921364761888981, -0.029053840786218643, -0.016548573970794678, 0.14776572585105896, 0.01718594878911972, -0.21374821662902832, 0.0925460010766983, 0.24666988849639893, 0.1848907619714737, -0.0033415351063013077, -0.07428636401891708, -0.8743060827255249, -0.029617469757795334, 0.0616220124065876, 1.311631441116333, -0.01727181114256382, 0.029871707782149315, -0.10310295969247818, -0.07513421773910522, -0.09347525984048843, -0.04317031428217888, 0.02400955930352211, 0.09685584157705307, -0.14351220428943634, 0.2755604386329651, 0.10728400945663452, 0.04675597324967384, -0.0716102197766304, -0.10464949905872345, 0.029193997383117676, -0.008684495463967323, 0.047399431467056274, -0.2427493780851364, 0.06502977013587952, -0.11381576210260391, 0.09393694251775742, 0.5771890878677368, 0.09792877733707428, -0.0430334210395813, -0.07128967344760895, 0.1172836571931839, 0.08906258642673492, 0.09564600139856339, -0.0814872458577156, 0.2708854675292969, 0.10321421921253204, -0.05546114593744278, 0.04767092317342758, -0.027838265523314476, 0.005628836341202259, -0.014475022442638874, -0.21389107406139374, -0.032700806856155396, 0.03300436958670616, -0.03501420095562935, 0.028333643451333046, -0.2880939245223999]]

A_text=[[-0.16105026006698608, -0.011110251769423485, 0.19474360346794128, 0.14129455387592316, 0.2676087021827698, 0.03914526477456093, 0.051934510469436646, 0.0029656575061380863, 0.06058579310774803, 0.020587623119354248, -0.07620168477296829, 0.0017512156628072262, -0.031059691682457924, 0.03328496590256691, 0.04912715032696724, 0.10031041502952576, -0.0006369666079990566, 0.16175402700901031, -0.9095277190208435, -0.0485018715262413, 0.42652246356010437, -0.03379426151514053, -0.11375755816698074, -0.08331144601106644, -0.07800639420747757, 0.0164011400192976, 0.01786782406270504, 0.002947648521512747, -0.9301432967185974, -0.8317728042602539, 0.05054181069135666, 0.10063590854406357, -0.11440923064947128, 0.006849635858088732, 0.05933069810271263, -0.0007061631768010557, 0.009704714640974998, -0.12621386349201202, 0.021004004403948784, 0.04991590604186058, -0.07231667637825012, -0.06852950900793076, -0.012227109633386135, -0.010848191566765308, -0.043312329798936844, 0.12133732438087463, 0.054895803332328796, 0.02028086967766285, -0.152424618601799, -0.06752902269363403, 0.014475339092314243, 0.06611117720603943, -0.0719716027379036, 0.06218452379107475, -0.0682876706123352, -0.24181245267391205, -0.06847988814115524, -0.2739875912666321, -0.004043693654239178, -0.05706389248371124, 0.10189157724380493, 0.018312029540538788, -0.9245225191116333, 1.0495437383651733, 0.07090707123279572, -0.08689094334840775, 0.01845768466591835, 0.17105640470981598, 0.16296517848968506, 0.33545929193496704, -0.062340281903743744, 0.11821921169757843, -0.08970364928245544, -0.13463090360164642, 0.03298032283782959, -0.011972043663263321, 0.02560706064105034, -0.017996355891227722, -0.010668531060218811, 0.03107372485101223, 0.12480566650629044, -0.09880151599645615, -0.02603950910270214, -0.12004052847623825, -0.0025171281304210424, -0.04889068007469177, 0.004912037402391434, -0.0068636732175946236, 0.041724223643541336, 0.03950532525777817, -0.08626094460487366, -0.22317416965961456, 0.21128197014331818, -0.0035547330044209957, 0.30905210971832275, 0.17277155816555023, 0.06103917956352234, -0.04155735671520233, 0.057673655450344086, -0.6860814690589905, 0.02525973692536354, -0.016479765996336937, -0.09824670106172562, 0.06767810881137848, -0.04915210232138634, -0.046514809131622314, -0.04862328618764877, -0.019410327076911926, 0.0026808620896190405, 0.11036130040884018, 0.08692502230405807, 0.015454322099685669, -0.16927483677864075, -0.12077300250530243, 0.06430499255657196, -0.09492117911577225, 0.12451404333114624, 0.08492594957351685, -0.10082435607910156, 0.17462919652462006, -0.13615252077579498, -0.13774190843105316, 0.09283896535634995, 0.05426250025629997, 0.03814012184739113, 0.03663138300180435, -0.04886918142437935, 0.6226625442504883, 0.13000425696372986, -0.30588674545288086, 0.12389856576919556, -0.14779409766197205, -0.23342303931713104, 0.09063517302274704, -0.056871406733989716, -0.0412011481821537, 0.002179223345592618, 0.17408819496631622, 0.04291342943906784, 0.16966702044010162, 0.1348450481891632, 0.07758409529924393, -0.18836823105812073, -0.4521143436431885, 0.03477559983730316, 0.36202794313430786, 0.06650681793689728, 0.051552385091781616, 0.0013721137074753642, 0.00031045646755956113, 0.10654403269290924, 0.054672203958034515, 0.14160946011543274, -0.08987165987491608, 0.07481890171766281, 0.04040186479687691, -0.2705269455909729, 1.0208780765533447, 0.0288397166877985, -0.04507508501410484, -0.1094658374786377, -0.06782108545303345, 0.044646162539720535, 1.1681162118911743, -0.09653088450431824, -0.5443752408027649, -0.06528891623020172, 0.1822851598262787, 0.012771312147378922, -0.02148411050438881, 0.04825528711080551, -0.026169128715991974, 0.1424134522676468, -0.13665084540843964, 0.18989090621471405, 0.0012267838465049863, -0.5550653338432312, -0.04810747504234314, -0.022171057760715485, 0.07446148246526718, 0.20225372910499573, -0.1265324354171753, 0.143564373254776, -0.060236457735300064, 0.08624690771102905, -0.012052606791257858, -0.07766597718000412, 0.013437393121421337, -0.20747089385986328, 0.07344315201044083, -0.2832103967666626, 0.19866053760051727, 0.2065349519252777, 0.04518812894821167, 0.009827183559536934, -0.027297090739011765, 0.003366764634847641, 0.05943872779607773, 0.04047395661473274, 0.09638887643814087, 0.010563243180513382, -0.04586523771286011, -0.005542443133890629, -0.06735154241323471, -0.1378321647644043, -0.16179093718528748, 0.2953992187976837, 0.08243168145418167, -0.06488988548517227, -0.10957270115613937, 0.034850336611270905, -0.03865223005414009, 0.03411014378070831, 0.04545148089528084, -0.10059104859828949, -0.014058534987270832, -0.05340004712343216, 0.1864192932844162, 0.20769579708576202, -0.030878392979502678, 0.03220069780945778, -0.31709083914756775, -0.08648308366537094, -1.695534110069275, -0.1958954930305481, 0.06839150190353394, -0.08092952519655228, 0.2509114742279053, -0.08950711786746979, 0.11196411401033401, -0.19079916179180145, -0.06625767052173615, 0.2974303364753723, 0.5474533438682556, -0.014875129796564579, 0.0893540158867836, 0.3171733021736145, -0.005388643126934767, -0.188052237033844, -0.09676861017942429, 0.12297647446393967, -0.014225331135094166, -0.04426676034927368, -0.029148366302251816, 0.06663838773965836, -0.3394853174686432, -0.009126115590333939, -0.13428935408592224, 0.02518254518508911, 1.245932698249817, -0.332650363445282, -0.17234928905963898, 0.12608930468559265, 0.029600225389003754, 0.06478645652532578, 0.05424913018941879, -0.4467238783836365, -0.024205245077610016, 0.0667078047990799, 0.03323345631361008, 0.18494142591953278, -0.1449659913778305, -0.1172422543168068, -0.05440263822674751, 0.034527864307165146, 0.03524988144636154, 0.24177689850330353, -0.046738214790821075, 0.01713244430720806, 0.042689137160778046, -0.09231965988874435, -0.012533756904304028, -0.11046484112739563, -0.04835275188088417, 0.2043789178133011, -0.01675703562796116, -0.04419010132551193, 0.0940142571926117, 0.03721897676587105, -0.10190192610025406, -0.053014710545539856, 0.0989437848329544, 0.1044960767030716, 0.025575241073966026, 0.16963300108909607, -0.013977698050439358, 0.13263079524040222, 0.09127150475978851, -0.1911265105009079, -0.044202160090208054, 0.020910926163196564, 0.09477987140417099, 0.2019660323858261, -0.008781367912888527, 0.5730989575386047, 0.029518360272049904, 0.17838580906391144, 0.22676226496696472, -0.07566428184509277, -0.1328761875629425, -0.22325493395328522, -0.07048814743757248, 0.028480658307671547, -0.0026588558685034513, -0.2921207845211029, -0.02739083766937256, 0.13997603952884674, -0.04744469374418259, -0.05231478065252304, -0.03989627957344055, -0.06168067082762718, -0.4136577844619751, -0.028954068198800087, -0.08510700613260269, -0.12031477689743042, -0.11329115182161331, -0.4501868784427643, 0.14278312027454376, 0.05349733307957649, -1.8525359630584717, -0.015990514308214188, 0.05508381500840187, -0.23890726268291473, 0.011022946797311306, -0.09695426374673843, 0.12564510107040405, 0.2947635054588318, 0.0012078712461516261, 0.08581730723381042, -0.2455296814441681, -0.06653011590242386, 0.07230022549629211, 0.030078815296292305, 0.08825719356536865, 0.132155179977417, 0.12917914986610413, 0.06773562729358673, -0.07604817301034927, -1.5802315473556519, -0.012937950901687145, -0.08247286826372147, 0.8637107014656067, -0.024589180946350098, -0.0032576690427958965, 0.10977067053318024, 0.0010635675862431526, 0.0509708896279335, 0.4018787443637848, 0.09928958117961884, 0.055960386991500854, -0.028923695906996727, 0.6715800166130066, 0.0754544660449028, 0.0476970337331295, 0.4984291195869446, -0.01568484492599964, -0.03266237676143646, -0.08187323808670044, 0.0838291347026825, -0.544268012046814, 0.02039981633424759, -0.07104165107011795, 0.01856783591210842, 1.529333472251892, 0.06750258803367615, 0.01904722861945629, -0.13991710543632507, 0.13790912926197052, 0.010385730303823948, 0.08475755155086517, 0.04974213242530823, 0.036461617797613144, 0.04395367205142975, -0.018217528238892555, -0.018443191424012184, -0.008939328603446484, 0.0018897666595876217, 0.1398943066596985, 0.06048549339175224, -0.028967561200261116, 0.11694078892469406, -0.07752068340778351, 0.03962619602680206, -0.08478676527738571],[-0.14549842476844788, 0.04309087246656418, 0.05238785594701767, -0.007347568403929472, 0.1540675163269043, 0.024982977658510208, 0.17123407125473022, -0.047120824456214905, 0.023867718875408173, 0.04586578533053398, -0.07306832820177078, 0.010814391076564789, 0.04004226624965668, 0.015251907519996166, 0.07173313200473785, 0.05971590057015419, 0.05007283762097359, 0.26494160294532776, -0.7858439683914185, -0.13636969029903412, 0.3513951897621155, -0.17690341174602509, -0.0373411662876606, -0.08554913103580475, -0.13766096532344818, 0.10878786444664001, 0.09611964225769043, -0.0801813155412674, -0.7861759662628174, -0.9330853223800659, 0.09836754202842712, 0.025373272597789764, -0.03258694335818291, 0.058591973036527634, -0.11223570257425308, 0.047303732484579086, -0.032734110951423645, -0.13897539675235748, 0.010085554793477058, 0.0003306558937765658, -0.09075236320495605, 0.03168000653386116, -0.11984995752573013, -0.06946342438459396, 0.030824758112430573, 0.006751204840838909, 0.029322948306798935, 0.038699522614479065, -0.1910925805568695, -0.10627032071352005, -0.030848076567053795, -0.052094507962465286, -0.07898057997226715, 0.09991732984781265, -0.050053466111421585, -0.08592841029167175, 0.07234888523817062, -0.16141797602176666, -0.04506848752498627, -0.04326605424284935, 0.12138336896896362, -0.03317001461982727, -0.9414033889770508, 1.124457836151123, 0.007058088202029467, -0.05081474035978317, 0.00539356516674161, 0.2602379620075226, 0.11191373318433762, 0.1538822054862976, -0.05855134502053261, 0.04509130120277405, -0.11358074098825455, -0.08131937682628632, 0.06539452075958252, 0.1165124699473381, -0.030833091586828232, -0.17742149531841278, 0.09778477996587753, 0.05859333276748657, 0.08060386031866074, -0.02486039139330387, -0.028118206188082695, -0.030277017503976822, 0.017206931486725807, 0.05906195193529129, -0.0730559229850769, 0.018980858847498894, -0.07716017961502075, -0.02148711122572422, -0.13176748156547546, -0.1356687992811203, 0.015263407491147518, 0.004339158069342375, 0.44620808959007263, 0.17749366164207458, 0.039584722369909286, -0.028914036229252815, -0.021745555102825165, -0.7584835886955261, -0.014802016317844391, -0.06218303367495537, 0.02222786843776703, -0.10887372493743896, -0.010053854435682297, -0.0925201028585434, -0.09913229197263718, -0.10739537328481674, 0.014755799435079098, 0.0716385766863823, 0.028330931439995766, -0.04082198813557625, -0.09544552117586136, -0.09613822400569916, 0.12497387081384659, -0.019188636913895607, -0.03782753646373749, 0.09795179963111877, -0.01980445347726345, 0.12521684169769287, -0.08948999643325806, -0.07336932420730591, 0.07655031234025955, 0.026506423950195312, 0.09840182960033417, -0.02995394729077816, -0.07505444437265396, 0.6008231043815613, 0.006002739071846008, -0.18421071767807007, 0.1812627762556076, -0.09794982522726059, -0.15887011587619781, 0.08259046822786331, 0.04676049202680588, -0.01616746559739113, -0.020652327686548233, 0.04298172891139984, -0.019279608502984047, 0.20618373155593872, 0.06380598247051239, 0.2170117348432541, -0.06782788038253784, -0.37688353657722473, 0.04159185662865639, 0.2893964648246765, 0.06608906388282776, -0.023579152300953865, 0.03340335562825203, 0.037082549184560776, 0.022900881245732307, -0.01360350102186203, 0.14583730697631836, -0.05020011588931084, 0.04015624523162842, 0.026206931099295616, -0.10384490340948105, 1.133067011833191, -0.12034977972507477, -0.07919654250144958, -0.013996549881994724, -0.055507220327854156, 0.033968593925237656, 1.203293800354004, -0.157659113407135, -0.617916464805603, -0.01827157661318779, 0.11631058156490326, 0.020536737516522408, -0.005608958192169666, 0.029857056215405464, -0.046932365745306015, 0.06903190910816193, -0.0778435543179512, 0.12389568239450455, -0.004816751927137375, -0.45453011989593506, -0.061062298715114594, 0.02091364935040474, -0.006310350727289915, 0.23282551765441895, -0.1026671752333641, 0.20952069759368896, -0.0746757760643959, 0.03213689103722572, 0.04643315449357033, 0.004936662968248129, 0.047559548169374466, -0.2102387547492981, 0.11895489692687988, -0.22412940859794617, 0.40541884303092957, 0.16672378778457642, -0.02689390629529953, -0.01625562645494938, -0.08531713485717773, 0.18970339000225067, -0.0020229965448379517, 0.011738951317965984, 0.07460609078407288, -0.055290915071964264, 0.04952413588762283, -0.05483856052160263, -0.02798786200582981, -0.21472710371017456, -0.1933019757270813, 0.17042040824890137, 0.09034719318151474, -0.08006449788808823, 0.008798740804195404, -0.09188418835401535, 0.06674917042255402, 0.03368816524744034, 0.08803901821374893, -0.0496441014111042, -0.06880921870470047, 0.003822837956249714, 0.18308429419994354, 0.13875851035118103, -0.0021772468462586403, -0.06828895211219788, -0.21477645635604858, -0.08496106415987015, -1.705343246459961, -0.16249150037765503, -0.0014280930627137423, -0.005060781259089708, 0.28453299403190613, -0.10942690074443817, 0.09202311933040619, -0.13424095511436462, -0.1459427922964096, 0.325349897146225, 0.5249423980712891, -0.17640985548496246, -0.016712410375475883, 0.2787122428417206, 0.03917420282959938, -0.22992201149463654, -0.1627522110939026, 0.006957662291824818, -0.029886849224567413, 0.024293232709169388, 0.006742776371538639, 0.142798513174057, -0.2159704864025116, -0.08365035057067871, -0.14272993803024292, 0.05394940823316574, 1.4683252573013306, -0.3654335141181946, -0.172387957572937, 0.2265254408121109, 0.071851447224617, 0.010819965042173862, -0.014594074338674545, -0.47599703073501587, 0.043519746512174606, 0.06168472394347191, 0.1023455262184143, 0.23031000792980194, -0.13880379498004913, 0.0075873336754739285, -0.0015827942406758666, 0.03505769371986389, 0.011172976344823837, 0.12656572461128235, 0.04988548904657364, 0.017573054879903793, 0.0035280296579003334, -0.10780762881040573, -0.043692175298929214, -0.05442763492465019, -0.03249384090304375, 0.17637678980827332, -0.056149594485759735, 0.017329443246126175, -0.08623402565717697, -0.05277702584862709, -0.040792521089315414, 0.05755874142050743, 0.14931735396385193, 0.10606175661087036, 0.06087258830666542, 0.1139073371887207, -0.10998260974884033, 0.1524120271205902, 0.10560242086648941, -0.22033950686454773, -0.0006932855467312038, 0.04602101817727089, 0.09394101053476334, 0.09442241489887238, -0.12097513675689697, 0.5240646600723267, 0.0799572616815567, 0.26289305090904236, 0.25184398889541626, -0.11211446672677994, -0.028037989512085915, -0.180324986577034, -0.0684136226773262, -0.011780256405472755, -0.030852066352963448, -0.21465842425823212, -0.07383936643600464, 0.18895667791366577, -0.0016504756640642881, -0.08116801828145981, -0.053717754781246185, -0.09731181710958481, -0.31454208493232727, -0.05025707930326462, -0.10465362668037415, -0.0827539786696434, -0.08275718986988068, -0.32008469104766846, 0.19324952363967896, 0.029142683371901512, -1.9650954008102417, 0.011547395028173923, 0.02958090975880623, -0.13339032232761383, -0.0035818477626889944, -0.0931791365146637, 0.10139387100934982, 0.15141944587230682, -0.03355497494339943, 0.06319902837276459, -0.3069559931755066, -0.08414170891046524, 0.04264133423566818, -0.07266277819871902, 0.14497502148151398, 0.21414414048194885, 0.23113100230693817, 0.03481950983405113, 0.03324795514345169, -1.7606868743896484, -0.005325273144990206, -0.0010467121610417962, 0.9743452668190002, -0.09106991440057755, 0.08040916174650192, 0.07036978751420975, -0.05776585265994072, 0.045182693749666214, 0.40762391686439514, 0.10099923610687256, -0.004484547767788172, -0.08604881167411804, 0.7319658994674683, 0.09251604229211807, 0.01888512820005417, 0.41528964042663574, 0.02507733553647995, 0.030699612572789192, -0.06961586326360703, 0.062260959297418594, -0.41313526034355164, 0.041599906980991364, -0.09358935803174973, -0.04775409400463104, 1.336578369140625, 0.09979918599128723, -0.04470101371407509, -0.03871769830584526, -0.005892172455787659, -0.008638707920908928, 0.12925222516059875, 0.014138600789010525, 0.039809226989746094, 0.0795636847615242, 0.016157476231455803, -0.013019563630223274, 0.08273754268884659, -0.008562729693949223, 0.06636219471693039, 0.028306035324931145, 0.04467454552650452, 0.08944692462682724, 0.10180919617414474, 0.13434462249279022, -0.15504322946071625],[-0.27511030435562134, 0.05865759402513504, 0.04279168322682381, 0.03128437325358391, 0.11815766245126724, -0.13554386794567108, 0.1928117573261261, 0.04699074849486351, -0.01152812223881483, -0.001567997271195054, -0.03866070881485939, 0.009659974835813046, 0.07334844768047333, 0.007728240918368101, 0.15918485820293427, 0.02190089039504528, 0.015536908991634846, 0.27026116847991943, -0.6269028186798096, -0.05530949681997299, 0.4642651081085205, -0.17109787464141846, -0.06428378820419312, -0.06770084798336029, -0.11987649649381638, 0.06424940377473831, 0.11316504329442978, -0.029658803716301918, -0.7175194025039673, -0.9268182516098022, 0.06752799451351166, 0.04112093150615692, -0.07153279334306717, 0.012646681629121304, -0.010072959586977959, 0.09902259707450867, -0.04118761420249939, -0.1380823254585266, -0.05591263622045517, 0.06868930906057358, -0.016303330659866333, 0.12608924508094788, -0.060102883726358414, -0.028824225068092346, 0.0672379806637764, -0.03432731330394745, 0.11042238026857376, 0.0070847878232598305, -0.07183565199375153, -0.16696715354919434, -0.024597501382231712, 0.025067510083317757, -0.09629471600055695, 0.10129330307245255, -0.12242162227630615, -0.07687826454639435, 0.10915283113718033, -0.1976689249277115, -0.00411358242854476, -0.03746778890490532, 0.11407570540904999, 0.01256844587624073, -0.9560040831565857, 0.9223133325576782, 0.10171865671873093, -0.06967589259147644, 0.09294509887695312, 0.1934785693883896, 0.11421850323677063, 0.35691389441490173, -0.07651706039905548, 0.034787166863679886, -0.05994967371225357, 0.025568660348653793, 0.07919221371412277, 0.09546723961830139, -0.052656568586826324, -0.17657509446144104, 0.0014977330574765801, 0.1102050319314003, -0.04205425828695297, -0.0601036362349987, -0.10477162897586823, -0.07280248403549194, 0.0581892654299736, 0.08567938208580017, -0.051982332020998, 0.06409324705600739, -0.09564915299415588, -0.07334109395742416, -0.09050242602825165, -0.14747385680675507, -0.06373976916074753, 0.051736440509557724, 0.47861021757125854, 0.18807841837406158, 0.10612975805997849, 0.004486746620386839, -0.005838420707732439, -0.7461854219436646, -0.002234020736068487, -0.07448825985193253, 0.08703796565532684, -0.047924019396305084, -0.03941207751631737, -0.021290449425578117, -0.10358305275440216, -0.08417955785989761, -0.007371671497821808, 0.03925604373216629, 0.06876812130212784, -0.01690259762108326, -0.007862416096031666, -0.09732786566019058, 0.0729689821600914, -0.10847540199756622, 0.026012366637587547, 0.04507535323500633, 0.019062811508774757, 0.07291487604379654, -0.08863385021686554, -0.06028342992067337, 0.016480715945363045, 0.03238143399357796, 0.06086039915680885, -0.0680687353014946, -0.10122501850128174, 0.49001750349998474, 0.052177730947732925, -0.09668873250484467, 0.1357744187116623, -0.2641780376434326, -0.07582690566778183, 0.10005903244018555, 0.07543201744556427, 0.00021815381478518248, -0.011005733162164688, 0.09911435842514038, 0.007495839614421129, 0.08219719678163528, 0.03409041091799736, 0.1039263978600502, 0.0011453487677499652, -0.41316738724708557, -0.043416574597358704, 0.16328012943267822, 0.0809258297085762, 0.034425657242536545, -0.0296615120023489, 0.025829415768384933, -0.028482362627983093, -0.050577208399772644, 0.09492858499288559, -0.0822846069931984, 0.054508525878190994, -0.0021502329036593437, -0.21903681755065918, 1.186640977859497, -0.16023112833499908, -0.08369995653629303, -0.0841996967792511, -0.014015784487128258, 0.055656157433986664, 1.2173501253128052, -0.14802035689353943, -0.601311445236206, 0.03597524017095566, 0.09704405069351196, 0.005569256842136383, -0.01167045533657074, 0.034219563007354736, -0.03923898935317993, 0.07449214160442352, -0.09660202264785767, 0.1369321048259735, 0.04780321195721626, -0.5084395408630371, -0.15297465026378632, 0.05302903428673744, -0.003930389415472746, 0.23875831067562103, -0.06007206812500954, 0.23618236184120178, -0.036956194788217545, -0.010925926268100739, -0.017802396789193153, 0.02912084199488163, 0.12217048555612564, -0.23828397691249847, 0.03397705405950546, -0.1775427609682083, 0.37972626090049744, 0.14126011729240417, -0.008490710519254208, -0.0187749695032835, -0.09811697900295258, 0.13556864857673645, 0.010379261337220669, -0.06200040504336357, 0.04480607807636261, 0.1368449181318283, 0.05225672200322151, -0.012038766406476498, -0.013815653510391712, -0.23236486315727234, -0.21917161345481873, 0.19113416969776154, 0.09319189190864563, -0.12262788414955139, -0.053591515868902206, -0.10184149444103241, -0.015583934262394905, -0.036359403282403946, 0.05072036758065224, -0.13049829006195068, -0.03961750492453575, -0.030747108161449432, 0.06774158775806427, 0.1411495953798294, 0.007617733906954527, 0.009074604138731956, -0.19812020659446716, -0.17230460047721863, -1.7041724920272827, -0.22472815215587616, 0.054414939135313034, 0.029879255220294, 0.31349143385887146, -0.11510992795228958, 0.0807470753788948, -0.1392066478729248, -0.17443569004535675, 0.20942124724388123, 0.480535089969635, -0.03987185284495354, 0.025480950251221657, 0.20311231911182404, -0.06112190708518028, -0.20461054146289825, -0.15305711328983307, 0.041922178119421005, -0.07005731761455536, 0.01218210719525814, 0.02031347155570984, 0.12615545094013214, -0.34850823879241943, -0.004334874451160431, -0.16456961631774902, 0.13808144629001617, 1.4155162572860718, -0.2548978924751282, -0.1733052134513855, 0.2638341784477234, 0.06502797454595566, 0.04098359867930412, 0.014351798221468925, -0.43848010897636414, -0.03475816547870636, 0.07663581520318985, 0.06120830774307251, 0.2219325453042984, -0.11178302019834518, 0.033220164477825165, -0.11440977454185486, 0.043844930827617645, 0.06216833367943764, 0.2514618933200836, 0.03940046578645706, 0.0156368650496006, -0.047918327152729034, -0.07178554683923721, 0.003325214609503746, -0.14235205948352814, -0.014638575725257397, 0.1278114765882492, -0.10834697633981705, 0.06657548993825912, -0.01911192014813423, -0.06985881924629211, -0.04738081991672516, -0.050244417041540146, 0.15208086371421814, 0.0551815927028656, -0.003607538528740406, 0.1313188225030899, -0.07786253094673157, 0.16361452639102936, 0.1918751746416092, -0.15448357164859772, -0.04380069300532341, -0.0331190824508667, 0.1312161237001419, -0.02380903623998165, -0.08304067701101303, 0.47850605845451355, 0.12272099405527115, 0.32983317971229553, 0.25371626019477844, -0.13512057065963745, -0.10650407522916794, -0.13987469673156738, -0.08349372446537018, 0.013895860873162746, 0.0574084036052227, -0.27634406089782715, 0.0731603354215622, 0.15266217291355133, -0.00823239330202341, -0.015497811138629913, -0.10587554425001144, -0.09994656592607498, -0.252330482006073, -0.014816607348620892, -0.15329676866531372, -0.02430025488138199, -0.03576129302382469, -0.20229600369930267, 0.17272089421749115, 0.06851766258478165, -1.9681167602539062, 0.025890592485666275, 0.01719648204743862, -0.3506562113761902, -0.010560308583080769, -0.05457199737429619, 0.07777997851371765, 0.16289429366588593, -0.09375747293233871, 0.09056788682937622, -0.2498246431350708, -0.07270738482475281, -0.0007234264630824327, -0.1356225162744522, 0.10403942316770554, 0.2135159820318222, 0.22887539863586426, 0.06422705948352814, -0.022575026378035545, -1.640804648399353, -0.01324460469186306, -0.024542611092329025, 1.0418232679367065, -0.05681566894054413, 0.0882493108510971, 0.013360279612243176, -0.02337593585252762, 0.10250219702720642, 0.36405688524246216, 0.13073478639125824, -0.0197578314691782, -0.10884726792573929, 0.6512574553489685, 0.1198728159070015, 0.06969296932220459, 0.4214216470718384, -0.007847163826227188, 0.0321865938603878, -0.042513180524110794, 0.08989019691944122, -0.2923957407474518, 0.03396959602832794, -0.17611707746982574, -0.03039509430527687, 1.383570909500122, 0.04413214698433876, 0.009767898358404636, -0.054858047515153885, 0.00830121524631977, 0.0669231042265892, 0.24779969453811646, -0.02091817930340767, 0.10341343283653259, 0.03688085451722145, -0.06727282702922821, -0.009728711098432541, -0.012193185277283192, -0.08192887902259827, 0.03745929151773453, 0.03554080054163933, 0.04307211935520172, 0.06475332379341125, 0.074927419424057, 0.09174244105815887, -0.020594893023371696]]

A_related=[[-0.056282125413417816, 0.05781134217977524, 0.20403604209423065, 0.10900914669036865, 0.258647084236145, 0.09625016152858734, 0.1691001057624817, -0.05324746295809746, -0.02907893992960453, 0.02559228613972664, -0.02406359277665615, -0.05841144919395447, -0.034747760742902756, -0.027355942875146866, 0.08161666244268417, 0.055801115930080414, 0.11988762021064758, 0.1725042462348938, -0.6020278930664062, -0.03158826008439064, 0.18589967489242554, -0.20386451482772827, -0.19656021893024445, -0.13804160058498383, -0.08594578504562378, 0.07050987333059311, 0.07169631868600845, -0.006833550054579973, -0.6240542531013489, -0.8700421452522278, 0.036983802914619446, -0.041118279099464417, -0.12819306552410126, 0.03471783176064491, -0.033384181559085846, -0.03151899576187134, 0.00279613072052598, -0.1542297899723053, -0.00771738076582551, 0.025026535615324974, -0.13868160545825958, -0.05099916458129883, -0.0923476591706276, 0.052969127893447876, -0.009627587161958218, 0.05402131751179695, 0.0958305075764656, 0.05757302790880203, -0.1295982301235199, -0.08487234264612198, 0.08188357204198837, 0.01812000200152397, -0.08406481891870499, 0.11455857753753662, -0.05651377514004707, -0.09858672320842743, 0.00690027792006731, -0.10814869403839111, -0.02458246983587742, -0.011279548518359661, 0.2148444652557373, -0.012022694572806358, -1.0425432920455933, 1.1320209503173828, 0.01922338455915451, -0.018346557393670082, -0.008023462258279324, 0.3428254723548889, 0.13948921859264374, 0.2016534060239792, -0.10037658363580704, 0.0630277618765831, -0.11514865607023239, -0.08461697399616241, 0.10754980146884918, -0.031112099066376686, 0.002556376624852419, -0.1143135353922844, 0.04701143875718117, 0.0516706146299839, 0.06726817786693573, 0.05056193843483925, -0.07794928550720215, -0.11177659779787064, -0.006966561544686556, -0.10133223980665207, -0.12227670848369598, 0.009665245190262794, -0.12945173680782318, 0.06700161844491959, -0.1038278192281723, -0.06781080365180969, 0.07697678357362747, -0.04560741409659386, 0.038036130368709564, 0.24513843655586243, 0.017609642818570137, -0.1623985916376114, -0.1712813675403595, -0.6132170557975769, 0.06304778158664703, -0.007531522307544947, 0.03212546184659004, 0.04688786342740059, -0.07220038026571274, -0.0065899621695280075, -0.13526380062103271, -0.15749068558216095, -0.07957285642623901, 0.09624949097633362, -0.028459109365940094, -0.0238628052175045, -0.04997440055012703, -0.1077406257390976, 0.07938171178102493, -0.12515592575073242, -0.009215053170919418, 0.1096220538020134, 0.001015380839817226, 0.10021933168172836, -0.10692987591028214, -0.10313745588064194, 0.1155388355255127, 0.04265325143933296, 0.022553512826561928, -0.10670676827430725, -0.008813424035906792, 0.5859136581420898, -0.05084351450204849, -0.18909770250320435, 0.18554578721523285, -0.18648487329483032, -0.1775844842195511, 0.03762969747185707, -0.015108784660696983, -0.007853837683796883, 0.07995671033859253, -0.035241056233644485, -0.014057706110179424, 0.36999574303627014, 0.11917107552289963, 0.2457914799451828, -0.08367802947759628, -0.4708893895149231, 0.07603374868631363, 0.5059947967529297, 0.13554446399211884, -0.006910995114594698, -0.027291065081954002, 0.07970008999109268, 0.08022844046354294, -0.019083639606833458, 0.1197386160492897, -0.15895922482013702, 0.031109709292650223, 0.05399608984589577, -0.14510615170001984, 1.261979579925537, -0.08503925800323486, -0.019289178773760796, -0.02054479531943798, -0.058087874203920364, -0.01402901578694582, 1.3841803073883057, -0.13977950811386108, -0.6689012050628662, -0.02814757637679577, 0.09123452007770538, -0.030433598905801773, -0.049004051834344864, -0.0016725059831514955, 0.023632366210222244, 0.10628454387187958, -0.0914529412984848, 0.22155015170574188, 0.06959029287099838, -0.5371829271316528, -0.022985875606536865, -0.08461453765630722, 0.020547736436128616, 0.20869238674640656, -0.07959045469760895, 0.1421007663011551, -0.061481911689043045, 0.15220554172992706, 0.02509344182908535, -0.11266102641820908, -0.0020377447362989187, -0.17832115292549133, 0.2542831003665924, -0.2290906012058258, 0.0944659635424614, 0.3127768933773041, -0.031501803547143936, 0.009264860302209854, -0.13564130663871765, 0.11574415862560272, -0.02063722535967827, 0.1109480932354927, 0.09958360344171524, 0.03460722789168358, 0.12133140116930008, -0.02905568666756153, -0.08827123045921326, -0.2344670444726944, -0.07136153429746628, 0.19879987835884094, 0.06270407140254974, -0.03812233358621597, -0.07854799181222916, -0.032822176814079285, 0.011708354577422142, 0.19349779188632965, 0.018929559737443924, -0.03567907586693764, -0.10362831503152847, 0.05343886837363243, 0.21108755469322205, 0.14659103751182556, 0.04056638479232788, -0.05601799860596657, -0.18518805503845215, -0.07925295829772949, -1.9454119205474854, -0.15837650001049042, -0.004962754435837269, -0.012562347576022148, 0.37157243490219116, -0.12566713988780975, 0.07491838932037354, -0.11660713702440262, -0.3051871359348297, 0.19230929017066956, 0.8111594319343567, -0.3034498989582062, 0.028186429291963577, 0.40656334161758423, 0.03771764039993286, -0.21745386719703674, -0.19534160196781158, 0.11051184684038162, -0.04674336314201355, 0.029611680656671524, -0.02749062329530716, 0.07248888909816742, -0.4719482958316803, 0.05168396607041359, -0.16529792547225952, 0.0289614200592041, 1.3327730894088745, -0.5096406936645508, -0.09800165891647339, 0.1383710354566574, 0.10406439751386642, -0.08394625782966614, -0.11144077032804489, -0.1429418921470642, -0.006741694174706936, 0.07037832587957382, -0.06165216863155365, 0.1924106776714325, -0.12561167776584625, -0.1037168875336647, -0.04906688258051872, 0.05163412541151047, -0.03828669711947441, 0.22299422323703766, 0.08341878652572632, -0.001144732697866857, 0.014210750348865986, -0.10820434987545013, -0.08704067766666412, -0.15773092210292816, -0.0724346712231636, 0.27907508611679077, 0.038809895515441895, -0.01903669163584709, -0.07441981136798859, -0.01344443578273058, 0.013005988672375679, 0.02234507165849209, 0.16176624596118927, 0.17016306519508362, 0.06119190901517868, 0.12723183631896973, -0.06535158306360245, 0.11709711700677872, 0.14730167388916016, -0.15560969710350037, -0.14551421999931335, 0.055965255945920944, 0.09113867580890656, 0.08816945552825928, -0.05482056364417076, 0.5949106216430664, 0.09750990569591522, 0.11769572645425797, 0.40964236855506897, -0.13397245109081268, -0.11925291270017624, -0.12787190079689026, 0.010844090953469276, -0.019810667261481285, 0.03764296695590019, -0.1758662611246109, -0.03404952958226204, 0.1434088796377182, 0.012416533194482327, -0.14471913874149323, -0.09138369560241699, -0.16686034202575684, -0.4046189486980438, -0.002954827155917883, -0.09796547889709473, -0.11460445076227188, -0.02104216068983078, -0.310818612575531, 0.1327546238899231, 0.06001432240009308, -2.1423139572143555, -0.04360301047563553, 0.05516465753316879, -0.0990949496626854, -0.05327335000038147, -0.14180141687393188, 0.10612037777900696, 0.36655399203300476, 0.051163408905267715, 0.051161762326955795, -0.08373113721609116, -0.02197236940264702, 0.07533402740955353, -0.03126285970211029, 0.13766278326511383, 0.2263992577791214, 0.12139121443033218, -0.001056260778568685, 0.01756037212908268, -1.752610683441162, 0.023434501141309738, 0.05486740916967392, 0.9901790618896484, -0.14460228383541107, 0.19907988607883453, 0.08826515078544617, -0.047077059745788574, -0.010605664923787117, 0.2717365324497223, 0.12656699120998383, 0.11117563396692276, -0.12209849804639816, 0.5927500128746033, 0.07356984168291092, 0.04945465549826622, 0.43550458550453186, 0.010780944488942623, 0.05168470740318298, -0.05574009567499161, 0.00251372461207211, -0.5390108227729797, 0.04825681075453758, -0.21589887142181396, -0.02234746515750885, 1.3701032400131226, 0.16089485585689545, -0.06496306508779526, -0.10746752470731735, 0.056042082607746124, -0.009388644248247147, 0.048808254301548004, 0.05609491094946861, -0.004081497434526682, 0.029003379866480827, 0.02800775319337845, -0.0778636485338211, 0.010386290028691292, -0.003288612235337496, 0.1332554817199707, 0.050991494208574295, -0.03160860762000084, 0.05064283311367035, 0.12602432072162628, 0.08584488183259964, -0.1331707388162613],[-0.09221743792295456, 0.13183623552322388, 0.19359879195690155, 0.11794282495975494, 0.29690465331077576, 0.051194269210100174, 0.2811954915523529, 0.11200488358736038, 0.035679686814546585, 0.026302427053451538, -0.05515040457248688, -0.036525797098875046, -0.06776336580514908, 0.039858367294073105, 0.07431094348430634, 0.13346606492996216, 0.07431075721979141, 0.22998106479644775, -0.7042140364646912, -0.02396903559565544, 0.4267963767051697, 0.03753206133842468, -0.12010553479194641, -0.08196064084768295, -0.0304811242967844, 0.02499409206211567, 0.04311786964535713, -0.03482295945286751, -0.6387984752655029, -0.8293545246124268, 0.08400297164916992, 0.05159462243318558, -0.1238444522023201, -0.02817729115486145, 0.017209507524967194, -0.1572418510913849, -0.06659253686666489, -0.11942336708307266, -0.028345542028546333, 0.028075216338038445, -0.11674784868955612, -0.11427374929189682, -0.11118956655263901, 0.04758775979280472, -0.0826636254787445, 0.07351343333721161, 0.11991629004478455, 0.055772069841623306, -0.0490921288728714, -0.131602942943573, 0.11442749202251434, -0.028173724189400673, -0.06428525596857071, 0.12162002921104431, -0.04555750638246536, -0.246665820479393, -0.023296674713492393, -0.2393840104341507, 0.008563539013266563, 0.0011100820265710354, 0.12938745319843292, -0.02143029309809208, -0.9621053338050842, 0.9239920973777771, 0.1865992695093155, -0.035095423460006714, 0.028299372643232346, 0.23156575858592987, 0.15106570720672607, 0.3949725031852722, -0.06577005237340927, 0.04817171022295952, -0.075813889503479, -0.09861692786216736, 0.07900813966989517, 0.03399214521050453, -0.010655401274561882, -0.217157244682312, 0.061125949025154114, 0.10167635977268219, 0.10405821353197098, -0.06523250043392181, -0.07934669405221939, -0.06952270865440369, 0.031314048916101456, -0.03567371517419815, -0.08142311871051788, 0.014480311423540115, -0.11708547174930573, 0.03713473677635193, -0.08402883261442184, -0.20370575785636902, 0.09470821917057037, -0.03697872534394264, 0.07974810153245926, 0.16946081817150116, 0.06367030739784241, -0.11047137528657913, -0.15252336859703064, -0.6033550500869751, 0.02422315813601017, -0.014661244116723537, -0.11849892139434814, 0.04390554502606392, -0.11108054965734482, -0.026714392006397247, -0.085232675075531, -0.08993785083293915, -0.012875575572252274, 0.16903409361839294, 0.04624663293361664, -0.045085009187459946, 0.05795971676707268, -0.10974452644586563, 0.12587858736515045, -0.19917993247509003, 0.1221139058470726, 0.09432698041200638, -0.12982727587223053, -0.0048785340040922165, -0.1466546356678009, -0.13661155104637146, 0.09244777262210846, 0.011659792624413967, -0.028894463554024696, -0.06307603418827057, -0.13247109949588776, 0.6421664357185364, 0.06767017394304276, -0.16381019353866577, 0.09115546941757202, -0.13356758654117584, -0.21348777413368225, 0.06688069552183151, -0.061059582978487015, -0.0639956071972847, 0.10221249610185623, 0.11212000995874405, -0.021514665335416794, 0.22641603648662567, 0.08244255185127258, 0.08419697731733322, -0.1579476296901703, -0.4181971251964569, -0.06815128028392792, 0.5382612347602844, 0.032383669167757034, 0.008849482983350754, -0.07480143755674362, 0.07312098145484924, 0.06396587193012238, 0.007928929291665554, 0.104604572057724, -0.1456131637096405, 0.05645013973116875, 0.0011316563468426466, -0.18020229041576385, 1.1886199712753296, 0.04997170716524124, -0.021465538069605827, -0.14327189326286316, -0.11026976257562637, 0.03670613467693329, 1.3356865644454956, -0.11411257833242416, -0.590133786201477, -0.09656164795160294, 0.05714627355337143, -0.05944405123591423, 0.0022147742565721273, 0.016675444319844246, 0.00612672558054328, 0.14310164749622345, -0.19404852390289307, 0.1828935593366623, 0.09326433390378952, -0.5682670474052429, 0.08142836391925812, -0.05288291722536087, -0.027292530983686447, 0.2339276522397995, -0.0944548025727272, 0.09276716411113739, -0.07068386673927307, 0.11667311936616898, 0.02882816642522812, -0.06469640880823135, 0.025148922577500343, -0.1348014771938324, 0.2937704920768738, -0.27036532759666443, 0.13235901296138763, 0.17438580095767975, 0.0254812091588974, -0.03118402697145939, -0.035578008741140366, 0.08324088156223297, -0.053045425564050674, 0.1178523525595665, 0.0866284891963005, 0.06239642947912216, 0.07743469625711441, -0.057060565799474716, -0.010426434688270092, -0.19032813608646393, -0.0671946257352829, 0.23578880727291107, 0.016877131536602974, -0.05170457065105438, -0.15643325448036194, -0.07796264439821243, -0.016866127029061317, 0.006633413955569267, 0.016153495758771896, -0.11700842529535294, -0.10158556699752808, -0.05349106714129448, 0.15870355069637299, 0.3232937455177307, -0.039838746190071106, -0.02700083889067173, -0.2802448570728302, -0.14252908527851105, -1.8827686309814453, -0.17098717391490936, 0.061201486736536026, -0.048127975314855576, 0.34305235743522644, -0.037841156125068665, 0.1365644335746765, -0.1862182915210724, -0.16200362145900726, 0.06047266721725464, 0.7141119241714478, -0.03868070989847183, 0.05602568760514259, 0.31430330872535706, 0.04538653418421745, -0.07435712963342667, -0.14128145575523376, 0.07994556427001953, -0.008028833195567131, -0.019051099196076393, 0.007893713191151619, 0.03567517176270485, -0.43884190917015076, 0.09252583235502243, -0.1776583045721054, -0.0077804019674658775, 1.3071808815002441, -0.4381943345069885, -0.12158315628767014, 0.2741274833679199, 0.0815189853310585, 0.04514984041452408, 0.008619766682386398, -0.23512327671051025, -0.011944809928536415, 0.0645415335893631, -0.08058923482894897, 0.2283603698015213, -0.19981342554092407, -0.18100771307945251, -0.21080216765403748, 0.07660336792469025, 0.0650191381573677, 0.405698299407959, 0.11253686249256134, 0.04340309649705887, 0.006193220149725676, -0.15823620557785034, -0.11736370623111725, -0.10040142387151718, -0.13195666670799255, 0.24178999662399292, 0.07811981439590454, -0.09124845266342163, 0.1426393985748291, 0.058727044612169266, -0.013665948994457722, 0.05034768953919411, 0.06915466487407684, 0.09860354661941528, 0.030743688344955444, 0.20883800089359283, 0.026277409866452217, 0.028648771345615387, 0.05162988230586052, -0.15821057558059692, -0.06789648532867432, 0.03845096379518509, 0.13425326347351074, 0.22133639454841614, -0.04643166437745094, 0.6235007643699646, 0.08272426575422287, 0.15611301362514496, 0.3226335942745209, -0.1628160923719406, -0.23002679646015167, -0.2163960337638855, -0.057509586215019226, -0.07071579247713089, 0.09415631741285324, -0.29451119899749756, 0.008642853237688541, 0.11214499175548553, 0.049115654081106186, -0.08796972036361694, -0.06209506466984749, -0.15738806128501892, -0.40114593505859375, -0.08962710946798325, -0.17875532805919647, -0.14310070872306824, 0.0007230710471048951, -0.36606165766716003, 0.27404841780662537, 0.08310467004776001, -2.0174400806427, -0.003933425527065992, -0.04714583605527878, -0.06064723804593086, -0.10876468569040298, -0.1033402532339096, 0.1797487884759903, 0.2718724012374878, 0.06790007650852203, 0.051339708268642426, -0.12107538431882858, -0.04895962402224541, 0.06894055753946304, -0.008661918342113495, 0.0638485923409462, 0.20351572334766388, 0.16166211664676666, 0.07392692565917969, 0.0069445897825062275, -1.6611672639846802, -0.048698823899030685, -0.08307921141386032, 0.923245370388031, -0.05252004414796829, 0.07755289226770401, 0.05947012081742287, 0.029619920998811722, -0.0022683602292090654, 0.2937573194503784, 0.1132836565375328, 0.1410476416349411, -0.08274829387664795, 0.7727604508399963, 0.0027627425733953714, 0.06347961723804474, 0.29681628942489624, 0.022011850029230118, 0.02518683485686779, -0.13512617349624634, 0.02393636479973793, -0.46840864419937134, -0.02434307150542736, -0.31609103083610535, 0.054059769958257675, 1.5131137371063232, 0.06843437999486923, -0.0738426223397255, -0.016264446079730988, 0.18199685215950012, -0.030786365270614624, 0.01503627561032772, 0.03773479536175728, 0.08292417228221893, 0.11297416687011719, -0.01127551682293415, -0.20889869332313538, 0.021722471341490746, -0.014084081165492535, 0.0977715253829956, 0.09163988381624222, -0.10753896087408066, 0.07353342324495316, 0.020170344039797783, 0.13756416738033295, -0.09542567282915115],[-0.14262133836746216, 0.17011485993862152, 0.19986940920352936, 0.03757215291261673, 0.25477883219718933, 0.061961542814970016, 0.21930217742919922, -0.06945288181304932, -0.08236297219991684, 0.036636754870414734, -0.03427779674530029, -0.01390316616743803, 0.05349583551287651, -0.03016447275876999, 0.1673709750175476, 0.10061508417129517, 0.08278605341911316, 0.3297921419143677, -0.6960071921348572, -0.11663448065519333, 0.30139002203941345, -0.2692094147205353, -0.11457719653844833, -0.12532959878444672, -0.07718873769044876, 0.025308862328529358, 0.13032537698745728, -0.04597723111510277, -0.6470173001289368, -0.9603588581085205, 0.12396950274705887, -0.0373685248196125, -0.032488737255334854, 0.006616803351789713, 0.02539127506315708, 0.06597723811864853, 0.001047622412443161, -0.12875331938266754, -0.07188788056373596, -0.011995033361017704, -0.12237279862165451, -0.07039368897676468, -0.1324593424797058, -0.04693330451846123, 0.03341686353087425, -0.0010337581625208259, 0.08325240761041641, 0.02756061591207981, -0.18774674832820892, -0.12075433135032654, 0.047110188752412796, -0.06593377143144608, -0.11045601218938828, 0.06385135650634766, -0.02505938895046711, -0.04751265421509743, 0.08421903848648071, -0.14065752923488617, 0.0035332266706973314, -0.07244812697172165, 0.21515488624572754, -0.0205821730196476, -0.8420597910881042, 1.02137291431427, 0.0632355734705925, 0.013955083675682545, -0.0184443648904562, 0.33106765151023865, 0.10817024856805801, 0.12321951240301132, -0.0939655601978302, -0.006582649890333414, 0.007003640290349722, -0.23751939833164215, 0.16736143827438354, 0.022235458716750145, -0.07131832093000412, -0.19707751274108887, 0.15087364614009857, 0.06515390425920486, 0.13261155784130096, 0.05701415240764618, -0.0698167011141777, -0.06428619474172592, 0.01684173382818699, -0.013200972229242325, -0.10191919654607773, 0.013037089258432388, -0.0739346370100975, 0.028706295415759087, -0.06418255716562271, -0.036735061556100845, 0.014299650676548481, -0.0721340999007225, -0.057362500578165054, 0.19970335066318512, 0.057697344571352005, -0.17062611877918243, -0.23909521102905273, -0.6277811527252197, -0.05181443691253662, -0.11670706421136856, 0.013391762971878052, -0.06827722489833832, 0.022925205528736115, -0.06715600937604904, -0.09877151250839233, -0.16853873431682587, -0.06726955622434616, 0.15205855667591095, -0.013493936508893967, -0.04756081476807594, -0.10288232564926147, -0.1305019110441208, 0.23869337141513824, -0.05683707818388939, 0.04143914952874184, 0.06903078407049179, -0.02735949121415615, -0.12748180329799652, -0.030246445909142494, -0.06736260652542114, 0.05866794288158417, 0.05171481892466545, 0.04090188816189766, -0.08559787273406982, 0.03424004092812538, 0.5651114583015442, -0.03703748434782028, -0.18275071680545807, 0.09798815846443176, -0.16723370552062988, -0.106397844851017, 0.04052582383155823, 0.03661307319998741, -0.043753478676080704, 0.013254129327833652, -0.03235853090882301, 0.009340123273432255, 0.21665692329406738, 0.02992781437933445, 0.2879928648471832, -0.088286854326725, -0.3901027739048004, -0.08785805851221085, 0.5576288104057312, 0.14250999689102173, -0.006254358682781458, -0.019872227683663368, 0.12662281095981598, -0.043555330485105515, -0.042589783668518066, 0.17224393784999847, -0.1171809509396553, 0.07450766861438751, 0.03428655490279198, -0.1365879625082016, 1.2238695621490479, -0.10487572103738785, -0.05478343740105629, 0.05402682349085808, -0.056142646819353104, -0.014755954034626484, 1.472480297088623, -0.16631557047367096, -0.7445419430732727, 0.003506345907226205, 0.11218663305044174, -0.09788769483566284, 0.02262302301824093, 0.06965415924787521, -0.0814538300037384, 0.15838421881198883, -0.08618980646133423, 0.3377113342285156, 0.06946059316396713, -0.502633273601532, -0.005300211254507303, -0.08127236366271973, -0.022371932864189148, 0.11622344702482224, -0.07832919806241989, 0.1758543699979782, -0.0834510400891304, 0.11191073805093765, 0.042381178587675095, -0.000767270743381232, 0.027487577870488167, -0.21553099155426025, 0.1016528308391571, -0.0495213158428669, 0.01137106865644455, 0.32349273562431335, -0.04330817982554436, 0.02827712893486023, -0.10535057634115219, 0.21858422458171844, -0.07949230074882507, 0.055892568081617355, 0.059877458959817886, 0.057163164019584656, 0.1681433767080307, -0.10747575759887695, -0.1402091532945633, -0.2002190202474594, -0.13132527470588684, 0.20562975108623505, 0.08427795022726059, -0.03506729379296303, -0.06940626353025436, -0.10767055302858353, 0.056990932673215866, 0.28675785660743713, 0.07377851009368896, -0.016370948404073715, -0.09397053718566895, 0.02643265388906002, 0.2187860757112503, 0.21167759597301483, 0.08029123395681381, -0.04363815858960152, -0.2547595798969269, -0.22649972140789032, -1.9065357446670532, -0.14748884737491608, -0.0850938931107521, -0.004572868347167969, 0.4526221454143524, -0.051739733666181564, 0.0856822058558464, -0.20344780385494232, -0.29937228560447693, 0.053346678614616394, 0.8952922821044922, -0.28480657935142517, 0.06610070914030075, 0.2849258482456207, 0.0345095731317997, -0.21704912185668945, -0.2559528052806854, 0.11796337366104126, -0.0815284475684166, -0.02483510412275791, 0.03342309966683388, 0.031684741377830505, -0.39552465081214905, -0.09433087706565857, -0.1422787755727768, 0.0679384246468544, 1.4353913068771362, -0.5311998128890991, -0.15542466938495636, 0.39426979422569275, 0.009518780745565891, -0.05230147764086723, -0.06092728301882744, -0.021769002079963684, 0.02980145625770092, 0.05663265287876129, -0.023902222514152527, 0.2770301401615143, -0.1658509075641632, -0.09848689287900925, 0.03945378586649895, 0.09031345695257187, -0.06422143429517746, 0.2231171578168869, 0.05634006857872009, -0.02266003005206585, 0.08248985558748245, -0.18207722902297974, -0.07352670282125473, -0.19593368470668793, -0.014025524258613586, 0.3152982294559479, 0.009579597972333431, -0.11002134531736374, -0.047789931297302246, -0.03635575249791145, -0.021478110924363136, 0.07072415947914124, 0.1797066330909729, 0.17097604274749756, 0.004141051322221756, 0.09640035778284073, -0.03632499650120735, 0.15184900164604187, 0.1233605444431305, -0.23810815811157227, -0.1375463753938675, 0.057575374841690063, 0.10719835758209229, 0.06202007830142975, -0.0910552367568016, 0.6091319918632507, 0.08718523383140564, 0.2255011647939682, 0.384980171918869, -0.11947494000196457, -0.1468566209077835, -0.10775923728942871, -0.0030696261674165726, -0.09642475098371506, 0.048563674092292786, -0.3416062891483307, -0.02288677729666233, 0.2166643738746643, -0.005396191030740738, -0.04446889087557793, -0.11905352026224136, -0.22574640810489655, -0.4485338628292084, -0.053340692073106766, -0.14243048429489136, -0.15503133833408356, 0.018089840188622475, -0.31531357765197754, 0.21062318980693817, 0.03959297016263008, -2.1343142986297607, -0.014547470025718212, 0.07671549916267395, -0.04121995344758034, -0.061352264136075974, -0.1639995574951172, 0.15527187287807465, 0.28800395131111145, -0.07317648082971573, 0.038983069360256195, -0.22650498151779175, -0.020917898043990135, 0.034370508044958115, -0.10593504458665848, 0.13732583820819855, 0.22048048675060272, 0.18226058781147003, 0.06794283539056778, 0.08754723519086838, -1.8438626527786255, -0.07888412475585938, 0.01912493072450161, 1.0255357027053833, -0.16556717455387115, 0.11023687571287155, 0.06497325003147125, 0.006258526351302862, -0.00916244275867939, 0.4129165709018707, 0.1147880032658577, 0.05978449061512947, -0.12383649498224258, 0.6876943707466125, 0.07944723218679428, 0.10306089371442795, 0.46558213233947754, 0.07623037695884705, -0.015583398751914501, -0.06635868549346924, 0.020844772458076477, -0.4176090657711029, 0.09277885407209396, -0.24297581613063812, 0.018782155588269234, 1.348115086555481, -0.07733359187841415, -0.053144421428442, 0.07875890284776688, 0.07852216809988022, 0.10114984959363937, 0.09094103425741196, 0.029254615306854248, -0.006585111375898123, 0.10634263604879379, 0.08121207356452942, -0.010975765995681286, 0.06586640328168869, -0.005057574715465307, 0.08384260535240173, 0.011438990943133831, 0.0363718643784523, -0.007401031907647848, 0.08568024635314941, 0.10832396894693375, -0.15109212696552277]]
for i in range(3):
    A_page[i]=get_embedding(page_title[i])
    A_text[i]=get_embedding(text_normal[i])
    A_related[i]=get_embedding(text_related[i])

    
k_limits=[10,50,100]
DATELOW=["2010-10-07","2012-05-21","2015-09-11"]
DATEHIGH=["2013-12-29","2018-07-14","2024-03-16"]
YEARL=["2010","2012","2015"]
YEARH=["2013","2018","2024"]
def normalize(A):
    cou=0
    for L in A:
        norm=0
        for i in L:
            norm+=i**2
        norm=norm**0.5
        for i in range(len(L)):
            A[cou][i]/=norm
        cou+=1    
    return A            
 
offsets=[40,50,100]        
        
page_distances_l2= [(50, 1.8837050225875553), (100, 1.9403044185510065), (150, 1.9708296675644617), (200, 1.9949545997100222), (250, 2.0191999565819736), (300, 2.045761154776524), (350, 2.058115257520464), (400, 2.0743942617526057), (450, 2.084043979239297), (500, 2.0932543723082246)]
page_distances_cos= [(50, 0.03486634375075015), (100, 0.036947155890254324), (150, 0.03807488533115), (200, 0.03908166488058118), (250, 0.040029407243831416), (300, 0.041035718145191136), (350, 0.04156756449733778), (400, 0.042220363777881964), (450, 0.04256959398237481), (500, 0.04310045694231501)]
page_distances_l2,page_distances_cos=page_embeddings_generation()

text_distances_l2= [(50, 1.8132955186538047), (100, 1.8864551304734367), (150, 1.9241086803679224), (200, 1.9518694694995247), (250, 1.978072362503776), (300, 1.9970392723749113), (350, 2.01324215585679), (400, 2.033064281402448), (450, 2.0445192481721075), (500, 2.0568740871616358)]
text_distances_cos= [(50, 0.05806634009685718), (100, 0.06266386869132246), (150, 0.0653575080812756), (200, 0.06734234839339981), (250, 0.06920985122530976), (300, 0.07066130581025953), (350, 0.07188655006678923), (400, 0.07305723420766086), (450, 0.07417033158768704), (500, 0.07497219700737334)]
related_distances_l2= [(50, 1.963955687999984), (100, 2.0273229344352273), (150, 2.076524390566289), (200, 2.11148090497903), (250, 2.13695439742632), (300, 2.159234676411386), (350, 2.17743021803941), (400, 2.191229665240798), (450, 2.204810289279476), (500, 2.217755806843175)]
related_distances_cos= [(50, 0.06152926509601153), (100, 0.06650869187526887), (150, 0.06968921145216289), (200, 0.0720463773458041), (250, 0.0739814214484823), (300, 0.07570209929955918), (350, 0.07701689380946286), (400, 0.07801281926749615), (450, 0.07901686913873596), (500, 0.07997174086526015)]
text_distances_l2,text_distances_cos,related_distances_l2,related_distances_cos=text_embeddings_generation()

for i in range(10):
    page_distances_l2[i]=(page_distances_l2[i][0],round(page_distances_l2[i][1],6))
    page_distances_cos[i]=(page_distances_cos[i][0],round(page_distances_cos[i][1],6))
    text_distances_l2[i]=(text_distances_l2[i][0],round(text_distances_l2[i][1],6))
    text_distances_cos[i]=(text_distances_cos[i][0],round(text_distances_cos[i][1],6))
    related_distances_l2[i]=(related_distances_l2[i][0],round(related_distances_l2[i][1],6))
    related_distances_cos[i]=(related_distances_cos[i][0],round(related_distances_cos[i][1],6))
    
page_dist={}
text_dist={}
page_dist_ranges={}
text_dist_ranges={}
page_distance_ranges_list={}
text_distance_ranges_list={}
related_dist={}
page_len_constraints=[50,100,300]
ranks_list=[[2*i-1 for i in range(1,5)],[5*i for i in range(1,11)],[10*i+1 for i in range(10)]]
page_dist["<->"]=[page_distances_l2[0][1]**2,page_distances_l2[1][1]**2,page_distances_l2[3][1]**2]
page_dist["<=>"]=[page_distances_cos[0][1],page_distances_cos[1][1],page_distances_cos[3][1]]
text_dist["<->"]=[text_distances_l2[0][1]**2,text_distances_l2[1][1]**2,text_distances_l2[3][1]**2]
text_dist["<=>"]=[text_distances_cos[0][1],text_distances_cos[1][1],text_distances_cos[3][1]]
page_dist_ranges["<->"]=[(page_distances_l2[0][1]**2,page_distances_l2[1][1]**2),(page_distances_l2[1][1]**2,page_distances_l2[3][1]**2),(page_distances_l2[3][1]**2,page_distances_l2[7][1]**2)]
page_dist_ranges["<=>"]=[(page_distances_cos[0][1],page_distances_cos[1][1]),(page_distances_cos[1][1],page_distances_cos[3][1]),(page_distances_cos[3][1],page_distances_cos[7][1])]
page_distance_ranges_list["<->"]=[[(page_distances_l2[0][1]**2,page_distances_l2[1][1]**2),(page_distances_l2[2][1]**2,page_distances_l2[3][1]**2)],[(page_distances_l2[1][1]**2,page_distances_l2[3][1]**2),(page_distances_l2[5][1]**2,page_distances_l2[7][1]**2)],[(page_distances_l2[1][1]**2,page_distances_l2[2][1]**2),(page_distances_l2[3][1]**2,page_distances_l2[4][1]**2),(page_distances_l2[5][1]**2,page_distances_l2[6][1]**2),(page_distances_l2[7][1]**2,page_distances_l2[8][1]**2)]]
page_distance_ranges_list["<=>"]=[[(page_distances_cos[0][1],page_distances_cos[1][1]),(page_distances_cos[2][1],page_distances_cos[3][1])],[(page_distances_cos[1][1],page_distances_cos[3][1]),(page_distances_cos[5][1],page_distances_cos[7][1])],[(page_distances_cos[1][1],page_distances_cos[2][1]),(page_distances_cos[3][1],page_distances_cos[4][1]),(page_distances_cos[5][1],page_distances_cos[6][1]),(page_distances_cos[7][1],page_distances_cos[8][1])]]
text_dist_ranges["<->"]=[(text_distances_l2[0][1]**2,text_distances_l2[1][1]**2),(text_distances_l2[1][1]**2,text_distances_l2[3][1]**2),(text_distances_l2[3][1]**2,text_distances_l2[7][1]**2)]
text_dist_ranges["<=>"]=[(text_distances_cos[0][1],text_distances_cos[1][1]),(text_distances_cos[1][1],text_distances_cos[3][1]),(text_distances_cos[3][1],text_distances_cos[7][1])]
text_distance_ranges_list["<->"]=[[(text_distances_l2[0][1]**2,text_distances_l2[1][1]**2),(text_distances_l2[2][1]**2,text_distances_l2[3][1]**2)],[(text_distances_l2[1][1]**2,text_distances_l2[3][1]**2),(text_distances_l2[5][1]**2,text_distances_l2[7][1]**2)],[(text_distances_l2[1][1]**2,text_distances_l2[2][1]**2),(text_distances_l2[3][1]**2,text_distances_l2[4][1]**2),(text_distances_l2[5][1]**2,text_distances_l2[6][1]**2),(text_distances_l2[7][1]**2,text_distances_l2[8][1]**2)]]
text_distance_ranges_list["<=>"]=[[(text_distances_cos[0][1],text_distances_cos[1][1]),(text_distances_cos[2][1],text_distances_cos[3][1])],[(text_distances_cos[1][1],text_distances_cos[3][1]),(text_distances_cos[5][1],text_distances_cos[7][1])],[(text_distances_cos[1][1],text_distances_cos[2][1]),(text_distances_cos[3][1],text_distances_cos[4][1]),(text_distances_cos[5][1],text_distances_cos[6][1]),(text_distances_cos[7][1],text_distances_cos[8][1])]]
related_dist["<->"]=[related_distances_l2[0][1]**2,related_distances_l2[1][1]**2,related_distances_l2[3][1]**2]
related_dist["<=>"]=[related_distances_cos[0][1],related_distances_cos[1][1],related_distances_cos[3][1]]



    
for (opn,op) in [("l2","<->"),("cos","<=>")]:
    if opn=="cos":
        A_text=normalize(A_text)
        A_page=normalize(A_page) 
        A_related=normalize(A_related)  
    q1()
    q2()
    q3()
    q4()
    q5()
    q6()
    q7()
    q8()
    q9()
    q10()
    q11()
    q12()
    q13()
    q14()
    q15()
    q16()
    q17()
    q18()
    q19()
    q20()
    q21()
    q22()
    q23()
    q24()
    q25()
    q26()
    q27()
    q28()
    q29()
    q30()
    q31()
    q32()
    q33()
    q34()
    q35()
    q36()
    q37()
    q38()
    q39()
    
    
    

