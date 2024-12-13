import os
import json
import re
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
from openai import OpenAI
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


class CombinedEvaluator:
    def __init__(self, results, questions=None, opm_elements=None, openai_api_key=None):
        self.questions = questions or []
        self.questions_by_id = {q["id"]: q for q in self.questions}
        self.results = results
        self.results_by_id = {r["id"]: r for r in self.results}
        self.opm_elements = self.load_opm_elements(opm_elements) if opm_elements else []
        self.eval_len = len(self.questions)

        # Initialize metrics
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bleurt = bleurt_score.BleurtScorer("BLEURT-20")
        self.client = OpenAI(api_key=openai_api_key)

    @staticmethod
    def load_opm_elements(opm_file):
        """
        Load OPM elements from the provided file, one element per line.
        """
        with open(opm_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    def match_opm_elements(self, text):
        """
        Find matched OPM elements in the text using exact matching.
        """
        matched_elements = {elem for elem in self.opm_elements if elem in text}
        return matched_elements

    def calculate_transparency_metrics(self, matched_prediction, matched_ground_truth):
        """
        Calculate precision, recall, and F1-score for transparency evaluation.
        """
        intersection = matched_prediction & matched_ground_truth
        intersection_count = len(intersection)

        precision = intersection_count / len(matched_prediction) if matched_prediction else 0.0
        recall = intersection_count / len(matched_ground_truth) if matched_ground_truth else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return precision, recall, f1_score, intersection

    @staticmethod
    def normalize_text(text):
        """
        Normalize text by lowercasing, removing punctuation, and lemmatizing.
        """
        text = re.sub(r"[^\w\s]", "", text.lower().strip())
        tokens = text.split()
        return " ".join(lemmatizer.lemmatize(token) for token in tokens if token not in stop_words)

    def score_loose_strict(self, reference, candidate, k=1.5):
        """
        Calculate Loose and Nonlinear Strict accuracy.
        Loose: Overlap proportion.
        Strict: Nonlinear function stricter than Loose.

        Args:
            reference (str): The reference answer.
            candidate (str): The candidate answer.
            k (int): Nonlinear penalty parameter (default=2).

        Returns:
            dict: Dictionary with loose and strict scores.
        """
        # Normalize and tokenize
        ref_tokens = set(self.normalize_text(reference).split())
        cand_tokens = set(self.normalize_text(candidate).split())

        # Calculate overlap
        overlap = len(ref_tokens & cand_tokens) / len(ref_tokens) if ref_tokens else 0.0

        # Calculate strict_score with nonlinear penalty
        strict_score = overlap**k

        return {"loose": overlap, "strict": strict_score}

    def score_rouge(self, reference, candidate):
        """
        Calculate ROUGE scores.
        """
        reference = self.normalize_text(reference)
        candidate = self.normalize_text(candidate)
        scores = self.rouge.score(reference, candidate)
        return {key: scores[key].fmeasure for key in ["rouge1", "rouge2", "rougeL"]}

    def score_bleurt(self, reference, candidate):
        """
        Calculate BLEURT score.
        """
        reference = self.normalize_text(reference)
        candidate = self.normalize_text(candidate)
        scores = self.bleurt.score(references=[reference], candidates=[candidate])
        return {"bleurt": scores[0]}

    def factuality_prompt(self, question, reference, answer):
        """
        Generate a factuality evaluation prompt for GPT Judge.
        """
        return (
            f"Given the following question and two answers, evaluate the factual consistency and coverage of the "
            f"Submitted Answer relative to the Reference Answer.\n\n"
            f"Question: {question}\n\n"
            f"Reference Answer: {reference}\n\n"
            f"Submitted Answer: {answer}\n\n"
            "Evaluate the Submitted Answer based on the following criteria:\n"
            "- Accuracy: Does the Submitted Answer correctly address the question?\n"
            "- Completeness: Does the Submitted Answer include all the key information present in the Reference Answer?\n"
            "- Consistency: Does the Submitted Answer contain any contradictory information compared to the Reference Answer?\n\n"
            "Please return a single grade based on the evaluation:\n"
            "- A: The Submitted Answer is a subset of the Reference Answer and is fully consistent.\n"
            "- B: The Submitted Answer is a superset of the Reference Answer and is fully consistent.\n"
            "- C: The Submitted Answer contains all the same details as the Reference Answer.\n"
            "- D: There is a disagreement between the Submitted Answer and the Reference Answer.\n"
            "- E: The Submitted Answer differs from the Reference Answer, but the differences do not matter for factuality.\n"
            "- F: The Submitted Answer does not address the question or is otherwise invalid.\n\n"
            "Only return one grade (A, B, C, D, E, or F)."
        )

    @staticmethod
    def grade_to_score(grade):
        """
        Convert GPT Judge grade to numeric score.
        """
        if grade in {"A", "B", "C", "E"}:
            return 1.0
        elif grade in {"D", "F"}:
            return 0.0
        else:
            raise ValueError(f"Invalid grade: {grade}")

    def score_gpt_judge(self, question, reference, answer):
        """
        Use GPT-4 model to calculate GPT Judge score.
        """
        prompt = self.factuality_prompt(question, reference, answer)
        completion = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a factual evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            top_p=1
        )
        gpt_output = completion.choices[0].message.content.strip()
        match = re.search(r'\b[A-F]\b', gpt_output)
        if match:
            grade = match.group(0)
            return self.grade_to_score(grade)
        else:
            raise ValueError(f"Invalid GPT output: {gpt_output}")

    def score(self, output_file):
        """
        Compute all metrics and save results.
        """
        total_scores = {
            "loose": 0.0,
            "strict": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleurt": 0.0,
            "gpt_judge": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        all_scores = []

        for question in tqdm(self.questions, desc="Evaluation Progress"):
            result = self.results_by_id.get(question["id"])
            if result:
                # Match OPM elements
                matched_ground_truth = self.match_opm_elements(question["answer"])
                matched_prediction = self.match_opm_elements(result["answer"])

                # Transparency metrics
                precision, recall, f1_score, intersection = self.calculate_transparency_metrics(
                    matched_prediction, matched_ground_truth
                )

                # BLEURT, ROUGE, Loose, Strict metrics
                loose_strict = self.score_loose_strict(question["answer"], result["answer"])
                rouge = self.score_rouge(question["answer"], result["answer"])
                bleurt = self.score_bleurt(question["answer"], result["answer"])

                # GPT Judge Score
                gpt_judge = self.score_gpt_judge(question["question"], question["answer"], result["answer"])

                # Aggregate metrics
                total_scores["loose"] += loose_strict["loose"]
                total_scores["strict"] += loose_strict["strict"]
                total_scores["rouge1"] += rouge["rouge1"]
                total_scores["rouge2"] += rouge["rouge2"]
                total_scores["rougeL"] += rouge["rougeL"]
                total_scores["bleurt"] += bleurt["bleurt"]
                total_scores["gpt_judge"] += gpt_judge
                total_scores["precision"] += precision
                total_scores["recall"] += recall
                total_scores["f1"] += f1_score

                # Store detailed results
                all_scores.append({
                    "id": question["id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "prediction": result["answer"],
                    "metrics": {
                        **loose_strict,
                        **rouge,
                        **bleurt,
                        "gpt_judge": gpt_judge,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1_score,
                        "matched_opm_elements_ground_truth": list(matched_ground_truth),
                        "matched_opm_elements_prediction": list(matched_prediction),
                        "matched_opm_elements_intersection": list(intersection)
                    },
                })

        # Calculate averages
        num_questions = len(all_scores)
        if num_questions > 0:
            for key in total_scores:
                total_scores[key] /= num_questions

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"overall": total_scores, "details": all_scores}, f, indent=2)

        print(f"\nEvaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA results for quality and transparency.")
    parser.add_argument("--ground_truth", required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--predictions", required=True, help="Path to the predicted answers JSON file.")
    parser.add_argument("--opmelem", required=True, help="Path to the file containing OPM elements, one per line.")
    parser.add_argument("--output", required=True, help="Path to save the evaluation JSON file.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OpenAI API key is not set in the environment variable 'OPENAI_API_KEY'.")

    with open(args.ground_truth, "r", encoding="utf-8") as gt_file:
        ground_truth = json.load(gt_file)

    with open(args.predictions, "r", encoding="utf-8") as pred_file:
        predictions = json.load(pred_file)

    evaluator = CombinedEvaluator(
        results=predictions,
        questions=ground_truth,
        opm_elements=args.opmelem,
        openai_api_key=api_key
    )
    evaluator.score(args.output)
