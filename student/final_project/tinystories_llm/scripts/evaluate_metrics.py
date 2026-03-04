import re
import json
import argparse
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def lexical_diversity_score(text):
    """
    Repetition score:
    1. Distinct-N: token diversity
    2. Consecutive duplicates: repeated words
    3. Burstiness: uneven token distribution
    4. Weighted: combine diversity and penalties
    """
    tokens = re.findall(r"\b\w+\b", text.lower())

    if len(tokens) < 4:
        return 1.0

    def distinct_n(tokens, n):
        if len(tokens) < n:
            return 1.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        ratio = len(set(ngrams)) / len(ngrams)
        return np.sqrt(ratio) 

    d1 = distinct_n(tokens, 1)
    d2 = distinct_n(tokens, 2)
    d3 = distinct_n(tokens, 3)

    dup = sum(tokens[i] == tokens[i+1] for i in range(len(tokens)-1))
    dup_rate = dup / len(tokens)
    consecutive_penalty = np.power(1 - dup_rate, 0.5)

    counts = np.array(list({t: tokens.count(t) for t in set(tokens)}.values()))
    burstiness = np.std(counts) / (np.mean(counts) + 1e-6)
    
    score = (
        0.3 * d1 + 
        0.2 * d2 + 
        0.1 * d3 + 
        0.4 * consecutive_penalty
    ) - 0.05 * burstiness

    return float(np.clip(score, 0, 1))

def entity_score(text, expected_entities):
    """
    Entity score:
    1. Active recall: model mention vs prompt mention
    2. Balance: variance of entity mentions
    3. Hallucination: detect unexpected internal capitals
    """
    if not expected_entities:
        return 1.0

    text_lower = text.lower()
    expected_set = [e.lower() for e in expected_entities]
    counts = {e: len(re.findall(rf'\b{re.escape(e)}\b', text_lower)) for e in expected_set}
    c_values = list(counts.values())

    basic_recall = sum(1 for c in c_values if c > 0) / len(expected_set)
    active_recall = sum(1 for c in c_values if c >= 2) / len(expected_set)
    recall_score = (basic_recall * 0.3) + (active_recall * 0.7)

    found_counts = [c for c in c_values if c > 0]
    if len(found_counts) > 1:
        cv = np.std(found_counts) / (np.mean(found_counts) + 1e-6)
        balance_score = max(0, 1 - cv)
    else:
        balance_score = 1.0 if len(found_counts) == len(expected_set) else 0.0

    text_no_starts = re.sub(r'(?:^|[.!?]\s+|["\']\s*)[A-Z][a-z]+', ' ', text)
    internal_caps = set(re.findall(r'\b[A-Z][a-z]+\b', text_no_starts))
    internal_caps = {w for w in internal_caps if w not in {'I', 'A'}}
    actual_names = {w.lower() for w in internal_caps}
    hallucinated_names = actual_names - set(expected_set)
    hal_penalty = len(hallucinated_names) * 0.5

    final_score = (recall_score * 0.6) + (balance_score * 0.4) - hal_penalty
    return float(np.clip(final_score, 0, 1))

def structure_score(text, target_sentences=4):
    """
    Structure score:
    1. Sentence count: match target
    2. Boundaries: check punctuation
    3. Completion: check ending
    4. Length: very short or long penalty
    """
    tokens = text.split()
    length = len(tokens)

    sentence_count = len(re.findall(r"[.!?]", text))
    boundary_error = 1 if sentence_count == 0 else 0
    incomplete = 0 if text.strip().endswith(('.', '!', '?')) else 1
    structure_penalty = abs(sentence_count - target_sentences) / max(1, target_sentences)
    length_penalty = 1 if length < 15 or length > 200 else 0

    score = 1 - (0.3*incomplete + 0.3*structure_penalty + 0.2*boundary_error + 0.2*length_penalty)
    return max(0, min(1, score))

def coherence_score(text):
    """
    Coherence score:
    1. Sentence similarity: cosine between embeddings
    2. Connectors: check frequency of logical words
    3. Length variance: check sentence length consistency
    """
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if len(sentences) < 2:
        return 1.0

    embeddings = model.encode(sentences)
    cos_vals = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]
    mean_cos = np.mean(cos_vals)
    cos_error = 1 - mean_cos

    connectors = ["because", "then", "so", "after", "but"]
    conn_count = sum(text.lower().count(c) for c in connectors)
    conn_freq = conn_count / max(1, len(sentences))
    conn_error = 1 - min(conn_freq, 1)

    sentence_lengths = [len(s.split()) for s in sentences]
    var_len = np.var(sentence_lengths) / 100

    score = 1 - (0.4*cos_error + 0.3*var_len + 0.3*conn_error)
    return max(0, min(1, score))

def compute_all_scores(text, expected_entities, target_sentences=4):
    return {
        "lexical diversity": lexical_diversity_score(text),
        "entity": entity_score(text, expected_entities),
        "structure": structure_score(text, target_sentences),
        "coherence": coherence_score(text)
    }

def evaluate_file(path, model_name):
    with open(path, "r") as f:
        outputs = json.load(f)

    all_scores = []
    for item in outputs:
        text = item["generated"]
        expected_entities = item.get("expected_entities", [])
        scores = compute_all_scores(text, expected_entities)
        item["scores"] = scores
        all_scores.append(scores)

    avg_scores = {k: np.mean([s[k] for s in all_scores]) for k in all_scores[0]}
    avg_scores["model"] = model_name
    return avg_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="List of json files")
    parser.add_argument("--names", nargs="+", required=True, help="Model names")
    parser.add_argument("--output", default="evaluation.csv")

    args = parser.parse_args()

    rows = [evaluate_file(p, n) for p, n in zip(args.inputs, args.names)]
    df = pd.DataFrame(rows).set_index("model")

    print(df)
    df.to_csv(args.output)

if __name__ == "__main__":
    main()