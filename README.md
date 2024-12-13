# 🤖Neuro-Conceptual Artificial Intelligence: Integrating OPM with Deep Learning to Enhance Question Answering Quality (COLING2025)
---
Based on our Paper - "Neuro-Conceptual Artificial Intelligence: Integrating OPM with Deep Learning to Enhance Question Answering Quality"
## ✨Introduction
We introduce Neuro-Conceptual Artificial Intelligence (NCAI, a specialization of the neuro-symbolic AI approach that integrates conceptual modeling using Object-Process Methodology (OPM) ISO 19450:2024 with deep learning to enhance question-answering quality.

By converting natural language text into OPM models using in-context learning, NCAI leverages the expressive power of OPM to represent complex processes and state changes that traditional triplet-based knowledge graphs cannot easily capture.

This rich structured knowledge representation improves reasoning transparency and answer accuracy in a question-answering system (OPM-QA).

<div align="center">
<img width="335" alt="overview_00(1)" src="https://github.com/user-attachments/assets/cace9937-c591-4a68-bcb5-7e20e73b1f34" />
</div>

---

## 🛠Requirement

Run the command to install the packages required.
```
pip install -r requirements.txt
```

---

## 📔File Structure

### Data
`k_opl.txt`: Knowledge base in Object-Process Language (OPL) format.

`k_nl.txt`: Knowledge base in Natural Language (NL) format.

`opm_elements.txt`: Object-Process Methodology (OPM) elements in the knowledge base.

`k_qa_pairs.json`: Example QA pairs used as context for generating answers.

`qa_pairs.json`: Ground truth QA pairs used for evaluation.

`q.json`: Questions extracted from qa_pairs.json.

### Scripts
`kbqa.py`: Implements the knowledge-based QA system.

- Uses domain knowledge (`k_opl.txt` or `k_nl.txt`) and example QA pairs (`k_qa_pairs.json`) to generate answers for questions in `q.json`.
  
- Outputs generated answers to a specified file (e.g., `result/answers.json`).

`evaluation.py`: Evaluates QA system outputs.

- Compares generated answers (`result/answers.json`) with ground truth (`qa_pairs.json`) using metrics like ROUGE, BLEURT, and GPT Judge.

- Outputs detailed evaluation results (e.g.,`result/evaluation.json`).
  
`evaluation_statistics.py`: Performs statistical analysis on evaluation results.

- Compares results from different experiments (e.g., OPL and NL knowledge bases).
- Outputs analysis to a specified file (e.g.,` result/statistics.txt`).
  
### Results
`result/answers.json`: Generated answers from the QA system.

`result/evaluation.json`: Evaluation results comparing generated answers with ground truth.

`result/statistics.txt`: Statistical comparison of evaluation results.

---

## 📜Quick start
Quick start: Using script file (`experiment.sh`)

```
>> bash experiment.sh
```
---
## 🤝 Cite
Please condiser citing this paper if you use the `code` or `data` from our work. Thanks a lot :)
```

```

