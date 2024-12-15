import json
import os
import numpy as np
from collections import defaultdict, Counter

# Step 1: Load data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Step 2: Preprocess text
def preprocess_text(text):
    # Tokenize and lower case the text
    return text.lower().split()

# Step 3: Build vocabulary and transition probabilities (Basic Model)
def build_model(data):
    transitions = defaultdict(Counter)

    conversations = data.get('dataset', {}).get('conversations', [])
    for conversation in conversations:
        utterances = conversation.get('utterances', [])
        for utterance in utterances:
            tokens = preprocess_text(utterance.get('utterance_text', ''))
            for i in range(len(tokens) - 1):
                transitions[tokens[i]][tokens[i + 1]] += 1

    # Convert counts to probabilities
    transition_probabilities = {
        word: {
            next_word: count / sum(counter.values())
            for next_word, count in counter.items()
        }
        for word, counter in transitions.items()
    }

    return transition_probabilities

# Step 4: Build n-gram model with smoothing (Improved Model)
def build_ngram_model(data, n=3):
    transitions = defaultdict(Counter)

    conversations = data.get('dataset', {}).get('conversations', [])
    for conversation in conversations:
        utterances = conversation.get('utterances', [])
        for utterance in utterances:
            tokens = preprocess_text(utterance.get('utterance_text', ''))
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n - 1])
                next_word = tokens[i + n - 1]
                transitions[ngram][next_word] += 1

    # Convert counts to probabilities with smoothing
    transition_probabilities = {}
    for ngram, counter in transitions.items():
        total_count = sum(counter.values())
        vocab_size = len(counter)
        transition_probabilities[ngram] = {
            next_word: (count + 1) / (total_count + vocab_size)  # Additive smoothing
            for next_word, count in counter.items()
        }

    return transition_probabilities

# Step 5: Combine JSON files from a directory
def combine_json_files(directory_path, output_filepath):
    combined_data = {"dataset": {"conversations": []}}  # Initialize with expected structure

    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "dataset" in data and "conversations" in data["dataset"]:
                    combined_data["dataset"]["conversations"].extend(data["dataset"]["conversations"])
                elif isinstance(data, list):
                    combined_data["dataset"]["conversations"].extend(data)
                else:
                    print(f"Skipping incompatible file format: {filename}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Combined JSON saved to {output_filepath}")

# Step 6: Generate text with temperature sampling (Improved Model)
def generate_text_with_temperature(model, start_words, length=10, temperature=1.0):
    current_ngram = tuple(start_words.split()[-2:])
    result = list(current_ngram)

    for _ in range(length - len(current_ngram)):
        if current_ngram not in model:
            break
        next_words, probabilities = zip(*model[current_ngram].items())

        # Apply temperature scaling
        scaled_probs = np.array(probabilities) ** (1 / temperature)
        scaled_probs /= scaled_probs.sum()

        next_word = np.random.choice(next_words, p=scaled_probs)
        result.append(next_word)
        current_ngram = tuple(result[-2:])

    return ' '.join(result)

# Step 7: Generate text based on the model (Basic Model)
def generate_text(model, start_word, length=10):
    current_word = start_word
    result = [current_word]

    for _ in range(length - 1):
        if current_word not in model:
            break
        next_words = list(model[current_word].keys())
        probabilities = list(model[current_word].values())
        current_word = np.random.choice(next_words, p=probabilities)
        result.append(current_word)

    return ' '.join(result)

# Step 8: Evaluate model
def evaluate_model(model, data, n=3):
    total_predictions = 0
    correct_predictions = 0
    log_prob_sum = 0
    word_count = 0

    conversations = data.get('dataset', {}).get('conversations', [])
    for conversation in conversations:
        utterances = conversation.get('utterances', [])
        for utterance in utterances:
            tokens = preprocess_text(utterance.get('utterance_text', ''))
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n - 1]) if n > 1 else tokens[i]
                next_word = tokens[i + n - 1]
                total_predictions += 1

                # Check if ngram exists in the model
                if ngram in model:
                    probabilities = model[ngram]
                    if next_word in probabilities:
                        correct_predictions += 1
                        log_prob_sum += np.log(probabilities[next_word])
                word_count += 1

    # Calculate metrics
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    perplexity = np.exp(-log_prob_sum / word_count) if word_count > 0 else float('inf')

    return {
        "precision": precision,
        "perplexity": perplexity
    }

# Step 9: Main execution
def main():
    directory_path = '/home/sunghun/바탕화면/ssu/Train/text_1'  
    output_filepath = 'train_data.json'  

    # Combine JSON files
    combine_json_files(directory_path, output_filepath)

    data = load_data(output_filepath)

    # Check environment variable to decide the model type
    model_type = os.getenv('MODEL_TYPE', 'basic')  # Default to 'basic'

    if model_type == 'basic':
        print("Running Basic Model...")
        model = build_model(data)
        metrics = evaluate_model(model, data, n=1)
    elif model_type == 'improved':
        print("Running Improved Model...")
        model = build_ngram_model(data, n=3)
        metrics = evaluate_model(model, data, n=3)
    else:
        print(f"Unknown MODEL_TYPE: {model_type}")
        return

    print("Evaluation Metrics:", metrics)

    # Generate text
    start_words = 'esg management' if model_type == 'improved' else 'esg'
    if model_type == 'improved':
        generated_text = generate_text_with_temperature(model, start_words, length=20, temperature=0.7)
    else:
        generated_text = generate_text(model, start_words, length=20)

    print("Generated Text:", generated_text)

if __name__ == '__main__':
    main()
