import math
import sys
def read_file_to_dict(filename):
    result = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            try:
                a, b = line.split(',', 1)
                key = a.strip()
                value = float(b.strip())
                result[key] = value
            except ValueError:
                continue  # Skip lines with non-numeric b
    return result

def calculate_rms_error(dict1, dict2):
    all_keys = set(dict1) | set(dict2)
    error_sum = 0.0
    for key in all_keys:
        v1 = dict1.get(key, 0.0)
        v2 = dict2.get(key, 0.0)
        error_sum += (v1 - v2) ** 2
        
    return math.sqrt(error_sum / len(all_keys)) if all_keys else float('nan')
location=sys.argv[1]
index_type=sys.argv[2]
metric_type=sys.argv[3]
query=sys.argv[4]
query_size=sys.argv[5]

# Replace these with your file paths
original_file = "brute_queries_output/q" + query +"/q" + query +"_output_" + metric_type +"_" + query_size +".txt"
if location=="postgres":
    calculated_file = location+"_queries_output/q" + query +"/q" + query +"_output_"  + metric_type +"_" + query_size +".txt"
else:
    calculated_file = location+"_queries_output/q" + query +"/q" + query +"_output_" + index_type + "_" + metric_type +"_" + query_size +".txt"
    
original_dict = read_file_to_dict(original_file)
calculated_dict = read_file_to_dict(calculated_file)

rms_error = calculate_rms_error(original_dict, calculated_dict)
with open("queries_accuracy_error", "a") as file:
        file.write("{}\n".format(rms_error))
print(f"RMS Error: {rms_error}")

