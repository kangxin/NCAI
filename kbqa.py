import os
import json
import argparse
from openai import OpenAI
import re
from tqdm import tqdm


def load_knowledge(file_path):
    """Load domain knowledge from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def load_example_qa(file_path):
    """Load example question-answer pairs from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_questions(file_path):
    """Load questions from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def clean_output(text):
    """Post-process the model's output to remove unwanted characters."""
    # Remove markdown format (like `**`) and newlines
    text = re.sub(r"(\*\*|`|_|\n)", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def knowledge_based_qa(client, domain_knowledge, example_qa_pairs, questions, model="gpt-4o-2024-11-20"):
    """
    Processes each question using domain knowledge and example QA pairs via OpenAI API.

    Args:
        client (OpenAI): The OpenAI client instance.
        domain_knowledge (str): Domain knowledge from a text file.
        example_qa_pairs (list): List of example QA pairs.
        questions (list): List of questions to ask the model.
        model (str): The OpenAI model to use (default: "gpt-4o-2024-11-20").

    Returns:
        list: A list of answers to the questions.
    """
    # Create the context
    context = f"Domain Knowledge:\n{domain_knowledge}\n\nExamples of Question-Answer Pairs:\n"
    for qa in example_qa_pairs:
        context += f"Q: {qa['question']}\nA: {qa['answer']}\n"

    answers = []
    for question in tqdm(questions, desc="QA Progress"):
        question_text = question["question"]
        # Refined system message without references to OPM
        system_message = (
            "You are a knowledgeable assistant. "
            "Your task is to answer questions based on the provided domain knowledge. "
            "Your answers should align closely with the domain knowledge, use precise terminology, and remain concise and accurate. "
            "Focus on identifying and describing key processes, objects, and states explicitly, and clarify their relationships where relevant."
        )
        context_with_question = context + f"\nNew Question:\nQ: {question_text}\nA (concise and precise):"

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": context_with_question}
            ],
            temperature=0,  # Deterministic output
            top_p=1
        )

        # Extract the answer and clean it
        raw_answer = response.choices[0].message.content.strip()
        clean_answer = clean_output(raw_answer)
        answers.append({"id": question["id"], "question": question_text, "answer": clean_answer})

    return answers


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Knowledge-Based QA System with OpenAI API.")
    parser.add_argument("--knowledge", required=True, help="Path to the knowledge file (text format).")
    parser.add_argument("--examples", required=True, help="Path to the example QA pairs file (JSON format).")
    parser.add_argument("--questions", required=True, help="Path to the questions file (JSON format).")
    parser.add_argument("--output", required=True, help="Path to the output file to save the answers (JSON format).")
    parser.add_argument("--model", default="gpt-4o-2024-11-20", help="OpenAI model to use (default: gpt-4o-2024-11-20).")
    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load files
    domain_knowledge = load_knowledge(args.knowledge)
    example_qa_pairs = load_example_qa(args.examples)
    questions = load_questions(args.questions)

    # Perform QA
    answers = knowledge_based_qa(client, domain_knowledge, example_qa_pairs, questions, model=args.model)

    # Save answers to the output file
    with open(args.output, 'w', encoding='utf-8') as output_file:
        json.dump(answers, output_file, ensure_ascii=False, indent=4)

    print(f"Answers have been saved to {args.output}")


if __name__ == "__main__":
    main()
