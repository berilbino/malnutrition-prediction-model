#!/usr/bin/env python
"""
HADOOP MAPREDUCE REDUCER
Aggregates Key-Value counts emitted by the Mapper after the Hadoop Shuffle & Sort phase.
Computes distributed regional health statistics across 100,000 records.
"""
import sys

def main():
    current_key = None
    current_count = 0

    print("Region\tMetric_Type\tCategory\tTotal_Count")
    print("-" * 65)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) != 4:
            continue

        region, metric_type, category, count_str = parts

        try:
            count = int(count_str)
        except ValueError:
            continue

        key = (region, metric_type, category)

        if current_key == key:
            current_count += count
        else:
            if current_key:
                r, m, c = current_key
                print(f"{r}\t{m}\t{c}\t{current_count:,}")
            current_key = key
            current_count = count

    # Output the last accumulated key
    if current_key:
        r, m, c = current_key
        print(f"{r}\t{m}\t{c}\t{current_count:,}")

if __name__ == '__main__':
    main()
