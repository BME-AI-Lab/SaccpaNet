import csv
import glob


def read_csv_to_dict(f):
    reader = csv.DictReader(f)
    list = []
    for rows in reader:
        list.append(rows)
    return list


if __name__ == "__main__":
    all_rows = []
    for i in glob.glob("**/log.csv", recursive=True):
        print(i)

        with open(i, "r") as f:
            all_rows += read_csv_to_dict(f)

    with open("aggregate_result.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
