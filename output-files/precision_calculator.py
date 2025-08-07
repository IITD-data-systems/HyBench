def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_integers(filepath):
    numbers = set()
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if is_integer(line):
                numbers.add(int(line))
    return numbers

def calculate_recall(original_file, calculated_file):
    original_numbers = read_integers(original_file)
    calculated_numbers = read_integers(calculated_file)

    if not original_numbers:
        print("No valid integers in original file.")
        return 0.0

    true_positives = original_numbers.intersection(calculated_numbers)
    recall = (len(true_positives) / len(original_numbers))*100
    if len(calculated_numbers)!=len(original_numbers):
        # print("a"*1000)
        pass
    return recall

# Example usage
import sys
location=sys.argv[1]
index_type=sys.argv[2]
metric_type=sys.argv[3]
query=sys.argv[4]
query_num=sys.argv[5]

original_file = "brute_queries_output/q" + query + "/q" + query + "_output_" + metric_type + "_" + query_num + ".txt"
if location=="postgres":
    calculated_file = location+"_queries_output/q" + query + "/q" + query + "_output_" + metric_type + "_" + query_num + ".txt"
else:
    calculated_file = location+"_queries_output/q" + query + "/q" + query + "_output_" + index_type + "_" + metric_type + "_" + query_num + ".txt"



recall = calculate_recall(original_file, calculated_file)
with open("queries_accuracy_error", "a") as file:
        file.write("{}\n".format(recall))
print(f"Recall: {recall:.4f}")

