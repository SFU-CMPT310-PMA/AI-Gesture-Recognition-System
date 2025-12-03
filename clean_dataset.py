import csv

input_file = "hand_gesture_dataset.csv"
output_file = "hand_gesture_dataset_clean.csv"

# Expected columns: 1 label + 21 landmarks × 3 coords = 64
expected_cols = 64

with open(input_file, newline="") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row_num, row in enumerate(reader, start=1):
        if len(row) == expected_cols:
            writer.writerow(row)
        else:
            print(f"Skipping malformed row {row_num}: found {len(row)} columns, expected {expected_cols}")

print(f"Cleaned dataset written to {output_file}")
