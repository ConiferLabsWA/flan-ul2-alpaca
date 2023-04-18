# Original Alpaca Dataset dataset located here: 
# https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json

import json
import csv
import random
from typing import List, Dict

def load_json_data(file_name: str) -> List[Dict]:
    with open(file_name, "r") as file:
        data = json.load(file)
    return data

def process_data(data: List[Dict]) -> List[Dict]:
    processed_data = []
    for item in data:
        question = f'{item["instruction"]}\n{item["input"]}'
        answer = item["output"]
        processed_data.append({"question": question, "answer": answer})
    return processed_data

def write_csv_data(file_name: str, data: List[Dict]):
    with open(file_name, "w", newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["question", "answer"])
        writer.writeheader()
        for item in data:
            writer.writerow(item)

def main():
    # Load and process JSON data
    json_data = load_json_data("alpaca_data_cleaned.json")
    processed_data = process_data(json_data)

    # Shuffle data
    random.shuffle(processed_data)

    # Calculate split indices
    total_count = len(processed_data)
    train_count = int(0.8 * total_count)

    # Split data
    train_data = processed_data[:train_count]
    test_data = processed_data[train_count:]

    # Write data to CSV files
    write_csv_data("data/train.csv", train_data)
    write_csv_data("data/test.csv", test_data)

if __name__ == "__main__":
    main()
