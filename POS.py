import collections
import numpy as np
import gensim.downloader as dl
from sklearn.metrics import precision_score
from transformers import RobertaTokenizerFast, RobertaModel
import faiss
import torch
import os

# Constants for model types
NOWORDVECTORS = 1
STATICWORDVECTORS = 2
CONTEXTUALWORDVECTORS = 3

def main():
    # File paths - modify these according to your directory structure
    train_file = 'pos/data/ass1-tagger-train'
    test_input_file = 'pos/data/ass1-tagger-dev-input'
    test_gold_file = 'pos/data/ass1-tagger-dev'

    # Ensure output directory exists
    os.makedirs('predictions', exist_ok=True)

    # Part 1.1: Baseline without vectors
    print("\n=== Running Part 1.1 (Baseline) ===")
    word_tag_map = train_word_counts(train_file)
    predict_baseline(word_tag_map, test_input_file, 'predictions/POS_preds_1.txt')
    evaluate_results(test_gold_file, 'predictions/POS_preds_1.txt')

    # Part 1.2: Static word vectors
    print("\n=== Running Part 1.2 (Static Vectors) ===")
    glove_model, static_data = train_static_model(train_file)
    predict_static(glove_model, static_data, test_input_file, 'predictions/POS_preds_2.txt')
    evaluate_results(test_gold_file, 'predictions/POS_preds_2.txt')

    # Part 1.3: Contextualized vectors
    print("\n=== Running Part 1.3 (Contextual Vectors) ===")
    roberta_model, context_data = train_contextual_model(train_file)
    predict_contextual(roberta_model, context_data, test_input_file, 'predictions/POS_preds_3.txt')
    evaluate_results(test_gold_file, 'predictions/POS_preds_3.txt')

def train_word_counts(train_file):
    """Part 1.1 training: simple frequency-based approach"""
    word_tag_map = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(train_file, 'r') as f:
        for line in f:
            for token in line.strip().split():
                word, tag = token.rsplit('/', 1)
                word_tag_map[word][tag] += 1
    return word_tag_map

def predict_baseline(word_tag_map, test_file, output_file):
    """Part 1.1 prediction"""
    with open(test_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            predicted = []
            for token in line.strip().split():
                word = token.rsplit('/', 1)[0]
                tag = max(word_tag_map.get(word, {'NN': 1}), key=word_tag_map.get(word, {'NN': 1}).get)
                predicted.append(f"{word}/{tag}")
            fout.write(' '.join(predicted) + '\n')

def train_static_model(train_file):
    """Part 1.2 training: GloVe vectors with similarity fallback"""
    # Load pre-trained vectors
    glove = dl.load("glove-wiki-gigaword-300")
    
    # Build training index
    train_words = []
    train_vectors = []
    train_tags = []
    word_tag_map = train_word_counts(train_file)
    
    for word in word_tag_map:
        if word in glove:
            train_words.append(word)
            train_vectors.append(glove[word])
            train_tags.append(max(word_tag_map[word], key=word_tag_map[word].get))
    
    # Convert to numpy arrays
    train_vectors = np.array(train_vectors).astype('float32')
    
    # Build FAISS index
    index = faiss.IndexFlatL2(train_vectors.shape[1])
    index.add(train_vectors)
    
    return glove, {
        'index': index,
        'words': train_words,
        'tags': train_tags,
        'word_tag_map': word_tag_map
    }

def predict_static(glove, static_data, test_file, output_file):
    """Part 1.2 prediction"""
    index = static_data['index']
    train_tags = static_data['tags']
    word_tag_map = static_data['word_tag_map']
    
    with open(test_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            predicted = []
            for token in line.strip().split():
                word = token.rsplit('/', 1)[0]
                
                # Check if word exists in training
                # 
                try:
                    # Find similar words using FAISS
                    vec = glove[word].reshape(1, -1).astype('float32')
                    _, I = index.search(vec, 1)
                    tag = train_tags[I[0][0]]
                except:
                    if word in word_tag_map:
                        tag = max(word_tag_map[word], key=word_tag_map[word].get)
                    else:
                        tag = 'NN'
                
                predicted.append(f"{word}/{tag}")
            fout.write(' '.join(predicted) + '\n')

def train_contextual_model(train_file):
    """Part 1.3 training: RoBERTa contextual embeddings"""
    # Use the fast tokenizer version
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    
    # Load model with ignore_mismatched_sizes to suppress warnings
    model = RobertaModel.from_pretrained('roberta-base', 
                                       ignore_mismatched_sizes=True)
    
    word_tag_map = train_word_counts(train_file)
    
    # Store all training embeddings and tags
    all_embeddings = []
    all_tags = []
    
    with open(train_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            words = [t.rsplit('/', 1)[0] for t in tokens]
            tags = [t.rsplit('/', 1)[1] for t in tokens]
            
            # Tokenize with return_offsets_mapping for better word alignment
            inputs = tokenizer(words, return_tensors='pt', 
                             is_split_into_words=True,
                             truncation=True,
                             return_offsets_mapping=True)
            
            with torch.no_grad():
                outputs = model(inputs.input_ids)
            
            # Get word IDs using fast tokenizer's word_ids() method
            word_ids = inputs.word_ids(batch_index=0)
            
            # Extract embeddings for each original word
            embeddings = []
            for word_idx in range(len(words)):
                # Find all tokens associated with current word
                token_indices = [i for i, wid in enumerate(word_ids) 
                               if wid == word_idx]
                
                if token_indices:
                    # Use first token's embedding
                    word_embedding = outputs.last_hidden_state[0, token_indices[0]].numpy()
                    embeddings.append(word_embedding)
            
            if embeddings:
                all_embeddings.extend(embeddings)
                all_tags.extend(tags)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(768)
    index.add(np.array(all_embeddings).astype('float32'))
    
    return (tokenizer, model), {
        'index': index,
        'tags': all_tags,
        'word_tag_map': word_tag_map
    }

def predict_contextual(models, context_data, test_file, output_file):
    """Part 1.3 prediction"""
    tokenizer, model = models
    index = context_data['index']
    tags = context_data['tags']
    word_tag_map = context_data['word_tag_map']
    
    with open(test_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            words = [t.rsplit('/', 1)[0] for t in line.strip().split()]
            predicted_tags = []
            
            inputs = tokenizer(words, return_tensors='pt', 
                             is_split_into_words=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process each word
            for word_idx in range(len(words)):
                word = words[word_idx]
                
                # First check if word exists in training
                if word in word_tag_map:
                    tag = max(word_tag_map[word], key=word_tag_map[word].get)
                else:
                    # Get contextual embedding
                    token_ids = [i for i, wid in enumerate(inputs.word_ids()) if wid == word_idx]
                    if token_ids:
                        embedding = outputs.last_hidden_state[0, token_ids[0]].numpy().astype('float32')
                        _, I = index.search(embedding.reshape(1, -1), 1)
                        tag = tags[I[0][0]]
                    else:
                        tag = 'NN'
                
                predicted_tags.append(f"{word}/{tag}")
            
            fout.write(' '.join(predicted_tags) + '\n')

def evaluate_results(gold_file, pred_file):
    """Calculate and print accuracy metrics"""
    y_true = []
    y_pred = []
    
    with open(gold_file, 'r') as f_gold, open(pred_file, 'r') as f_pred:
        for gold_line, pred_line in zip(f_gold, f_pred):
            gold_tokens = gold_line.strip().split()
            pred_tokens = pred_line.strip().split()
            
            for gold_token, pred_token in zip(gold_tokens, pred_tokens):
                _, true_tag = gold_token.rsplit('/', 1)
                _, pred_tag = pred_token.rsplit('/', 1)
                y_true.append(true_tag)
                y_pred.append(pred_tag)
    
    accuracy = precision_score(y_true, y_pred, average='micro')
    print(f"Accuracy: {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()