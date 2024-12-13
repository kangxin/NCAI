#! /bin/bash

data_dir="../data"
result_dir="../result"

# Step 1.
python kbqa.py --knowledge ${data_dir}/k_opl.txt --examples ${data_dir}/k_qa_pairs.json --questions ${data_dir}/q.json --output ${result_dir}/a_opl.json

# Step 2.
python evaluation.py --ground_truth ${data_dir}/qa_pairs.json --predictions ${result_dir}/a_opl.json --opmelem ${data_dir}/opm_elements.txt --output ${result_dir}/evaluation_opl.json

# Step 3.
python kbqa.py --knowledge ${data_dir}/k_nl.txt --examples ${data_dir}/k_qa_pairs.json --questions ${data_dir}/q.json --output ${result_dir}/a_nl.json

# Step 4.
python evaluation.py --ground_truth ${data_dir}/qa_pairs.json --predictions ${result_dir}/a_nl.json --opmelem ${data_dir}/opm_elements.txt --output ${result_dir}/evaluation_nl.json

# Step 5.
python evaluation_statistics.py --eval_opl ${result_dir}/evaluation_opl.json --eval_nl  ${result_dir}/evaluation_nl.json --output ${result_dir}/evaluation_statistics.txt
