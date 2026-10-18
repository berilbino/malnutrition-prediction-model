"""
HADOOP MAPREDUCE EXECUTOR & SIMULATOR
Executes the MapReduce pipeline:
Input CSV -> Mapper -> In-Memory Sort / Shuffle -> Reducer -> Output Report
Also provides the exact command for submitting to a multi-node Apache Hadoop cluster.
"""
import subprocess
import os
import sys

def run_hadoop_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    input_csv = os.path.join(project_dir, 'data', 'unicef_labeled_data.csv')
    mapper_script = os.path.join(base_dir, 'mapper.py')
    reducer_script = os.path.join(base_dir, 'reducer.py')
    output_file = os.path.join(base_dir, 'hadoop_output_report.txt')

    print("=" * 70)
    print("RUNNING HADOOP MAPREDUCE JOB (SIMULATION ON 100,000 UNICEF RECORDS)")
    print("=" * 70)
    print(f"HDFS Input Path Equivalent : {input_csv}")
    print(f"Mapper Script              : {mapper_script}")
    print(f"Reducer Script             : {reducer_script}")
    print(f"HDFS Output Equivalent     : {output_file}")
    print("-" * 70)

    # 1. Step 1: Run Mapper on input data
    print(">>> Phase 1: Mapper Streaming Started (Mapping records)...")
    with open(input_csv, 'r', encoding='utf-8') as infile:
        p_map = subprocess.Popen(
            [sys.executable, mapper_script],
            stdin=infile,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        map_out, map_err = p_map.communicate()

    if p_map.returncode != 0:
        print("Mapper Error:", map_err)
        return

    # 2. Step 2: Hadoop Shuffle & Sort Phase
    print(">>> Phase 2: Hadoop Shuffle & Sort Phase (Grouping by Key)...")
    lines = [l for l in map_out.strip().split('\n') if l]
    sorted_lines = sorted(lines)
    sorted_text = '\n'.join(sorted_lines) + '\n'

    # 3. Step 3: Run Reducer on sorted pairs
    print(">>> Phase 3: Reducer Aggregation Phase (Calculating distributions)...")
    p_red = subprocess.Popen(
        [sys.executable, reducer_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    red_out, red_err = p_red.communicate(input=sorted_text)

    if p_red.returncode != 0:
        print("Reducer Error:", red_err)
        return

    # Save output
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(red_out)

    print(">>> MapReduce Job Successfully Completed!")
    print(f">>> Output saved to: {output_file}\n")
    print("Sample MapReduce Output:")
    print("-" * 65)
    print('\n'.join(red_out.split('\n')[:20]))

    # Print real Hadoop cluster command for their documentation/viva
    print("\n" + "=" * 70)
    print("OFFICIAL APACHE HADOOP CLUSTER SUBMISSION COMMAND (FOR YOUR PROJECT REPORT):")
    print("=" * 70)
    cluster_cmd = (
        "hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\\n"
        "  -input /user/hadoop/unicef/unicef_labeled_data.csv \\\n"
        "  -output /user/hadoop/unicef/output_report \\\n"
        "  -mapper mapper.py \\\n"
        "  -reducer reducer.py \\\n"
        "  -file mapper.py \\\n"
        "  -file reducer.py"
    )
    print(cluster_cmd)
    print("=" * 70)

if __name__ == '__main__':
    run_hadoop_simulation()
