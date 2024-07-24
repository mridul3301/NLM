import argparse
import pandas as pd
from utils.file_utils import get_filenames

def generate_text_file(csv_path, csv_file_list, destination_path):
    for csv_file in csv_file_list:
        try:
            data = pd.read_csv(f"{csv_path}/{csv_file}")
            articles = data['Article'].str.cat(sep=' ')
            destination = f"{destination_path}/{csv_file.replace('.csv', '.txt')}"
            print(f"Writing to {destination}")
            with open(destination, "w", encoding='utf-8') as f:
                f.write(articles)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Convert CSV files to text files in UTF-8 format.")
        parser.add_argument("--csv_data_path", type=str, required=True, help="Path to the CSV data directory.")
        parser.add_argument("--text_data_path", type=str, required=True, help="Path to save the text files.")

        args = parser.parse_args()

        csv_files = get_filenames(args.csv_data_path)
        generate_text_file(args.csv_data_path, csv_files, args.text_data_path)
