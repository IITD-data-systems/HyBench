from collections import defaultdict
import sys

def parse_file(filepath):
    year_to_ids = defaultdict(set)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            year, id_, _ = parts
            year_to_ids[int(year)].add(id_)
    return year_to_ids

def parse_file1(filepath):
    year_to_ids = defaultdict(set)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            year, id_, _ = parts
            year_to_ids[int(year)].add(id_)
    return year_to_ids


def compute_average_recall(original_map, calculated_map):
    recalls = []
    for year in original_map:
        orig_ids = original_map[year]
        calc_ids = calculated_map.get(year, set())
        if not orig_ids:
            continue
        correct = orig_ids & calc_ids
        recall = len(correct) / len(orig_ids)
        recalls.append(recall)
    return (sum(recalls) / len(recalls))*100 if recalls else 0.0

if __name__ == "__main__":
    location=sys.argv[1]
    index_type=sys.argv[2]
    metric_type=sys.argv[3]
    query_number=sys.argv[4]
    query_size=sys.argv[5]
    original_file = "brute_queries_output/q"+ query_number +"/q"+ query_number +"_output_" + metric_type + "_" + query_size + ".txt"      # replace with your actual file path
    calculated_file = location+"_queries_output/q"+ query_number +"/q"+ query_number +"_output_" + metric_type + "_" + query_size + ".txt"  # replace with your actual file path

    original_map = parse_file(original_file)
    calculated_map = parse_file1(calculated_file)
    # print(calculated_map)
    average_recall = compute_average_recall(original_map, calculated_map)
    with open("queries_accuracy_error", "a") as file:
        file.write("{}\n".format(average_recall))

    print(f"Average Recall: {average_recall:.4f}")
