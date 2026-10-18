#!/usr/bin/env python
"""
HADOOP MAPREDUCE MAPPER
Processes UNICEF records line by line from HDFS / sys.stdin.
Emits Key-Value pairs:
  Key: Region \t Target_Type \t Category
  Value: 1
"""
import sys

def main():
    # Read line by line from standard input (streaming from HDFS)
    first_line = True
    header_indices = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        
        # Read and map header line
        if first_line:
            header_indices = {name: idx for idx, name in enumerate(parts)}
            first_line = False
            continue

        try:
            region = parts[header_indices['Region']]
            survival = parts[header_indices['Survival_Label']]
            malnutrition = parts[header_indices['Malnutrition_Label']]
            disease = parts[header_indices['Disease_Label']]

            # Emit Key-Value pairs (Tab-separated as required by Hadoop)
            # 1. Regional Survival Counts
            print(f"{region}\tSurvival\t{survival}\t1")
            
            # 2. Regional Malnutrition Counts
            print(f"{region}\tMalnutrition\t{malnutrition}\t1")
            
            # 3. Regional Disease Counts
            print(f"{region}\tDisease\t{disease}\t1")

        except (IndexError, KeyError):
            continue

if __name__ == '__main__':
    main()
