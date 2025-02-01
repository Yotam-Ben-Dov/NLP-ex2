import collections
from sklearn.metrics import precision_score
import numpy
import gensim.downloader as dl
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors

import collections
import numpy as np
import gensim.downloader as dl
from sklearn.metrics import precision_score
from transformers import RobertaTokenizer, RobertaModel
import faiss
import torch

# Constants
NOWORDVECTORS = 1
STATICWORDVECTORS = 2
CONTEXTUALWORDVECTORS = 3

def main():
    # File paths
    train_file = 'pos/data/ass1-tagger-train'
    test_file = 'pos/data/ass1-tagger-dev-input'
    test_output_file = 'pos/data/ass1-tagger-dev'

    # Part 1.1: No word vectors
    print("Running Part 1.1 (No Word Vectors)...")
    word_tag_counts = train_pos_model(train_file, NOWORDVECTORS)
    predict_and_evaluate(test_file, test_output_file, NOWORDVECTORS, word_tag_counts)

    # Part 1.2: Static word vectors
    print("\nRunning Part 1.2 (Static Word Vectors)...")
    static_model = train_pos_model(train_file, STATICWORDVECTORS)
    predict_and_evaluate(test_file, test_output_file, STATICWORDVECTORS, static_model)

    # Part 1.3: Contextualized vectors
    print("\nRunning Part 1.3 (Contextualized Vectors)...")
    context_model = train_pos_model(train_file, CONTEXTUALWORDVECTORS)
    predict_and_evaluate(test_file, test_output_file, CONTEXTUALWORDVECTORS, context_model)

def train_pos_model(train_file, model_type):
    if model_type == NOWORDVECTORS:
        return train_word_counts(train_file)
    
    elif model_type == STATICWORDVECTORS:
        glove_model = dl.load("glove-wiki-gigaword-200")
        word_tag_counts = train_word_counts(train_file)
        return {
            'glove': glove_model,
            'word_tags': word_tag_counts,
            'train_vectors': [],
            'train_tags': []
        }
    
    elif model_type == CONTEXTUALWORDVECTORS:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        word_tag_counts = train_word_counts(train_file)
        
        # Store training embeddings
        index = faiss.IndexFlatL2(768)
        tag_list = []
        
        with open(train_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                words = [t.rsplit('/', 1)[0] for t in tokens]
                tags = [t.rsplit('/', 1)[1] for t in tokens]
                
                inputs = tokenizer(words, is_split_into_words=True, 
                                 return_tensors='pt', truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                word_embeddings = []
                for word_idx in range(len(words)):
                    token_indices = [i for i, wid in enumerate(inputs.word_ids()) 
                                   if wid == word_idx]
                    if token_indices:
                        embedding = outputs.last_hidden_state[0, token_indices[0]].numpy()
                        word_embeddings.append(embedding)
                
                index.add(np.array(word_embeddings).astype('float32'))
                tag_list.extend(tags)
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'index': index,
            'tags': tag_list,
            'word_tags': word_tag_counts
        }

def predict_and_evaluate(test_file, gold_file, model_type, model):
    pred_file = f'POS_preds_{model_type}.txt'
    
    if model_type == NOWORDVECTORS:
        predict_wordcount(model, test_file, pred_file)
    
    elif model_type == STATICWORDVECTORS:
        predict_static(model, test_file, pred_file)
    
    elif model_type == CONTEXTUALWORDVECTORS:
        predict_contextual(model, test_file, pred_file)
    
    # Evaluate results
    compare_results(gold_file, pred_file)

def train_word_counts(train_file):
    word_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(train_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                word, tag = token.rsplit('/', 1)
                word_tag_counts[word][tag] += 1
    return word_tag_counts

def predict_wordcount(word_tag_counts, test_file, output_file):
    with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            tokens = line.strip().split()
            predicted = []
            for token in tokens:
                word = token.rsplit('/', 1)[0]
                if word in word_tag_counts:
                    tag = max(word_tag_counts[word], key=word_tag_counts[word].get)
                else:
                    tag = 'NN'
                predicted.append(f"{word}/{tag}")
            out_f.write(' '.join(predicted) + '\n')

def predict_static(model_data, test_file, output_file):
    glove = model_data['glove']
    word_tags = model_data['word_tags']
    
    with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            tokens = line.strip().split()
            predicted = []
            for token in tokens:
                word = token.rsplit('/', 1)[0]
                
                if word in word_tags:
                    tag = max(word_tags[word], key=word_tags[word].get)
                elif word in glove:
                    # Find most similar word in training data
                    sims = []
                    for train_word in word_tags:
                        if train_word in glove:
                            sim = np.dot(glove[word], glove[train_word])
                            sims.append((sim, train_word))
                    if sims:
                        _, best_word = max(sims)
                        tag = max(word_tags[best_word], key=word_tags[best_word].get)
                    else:
                        tag = 'NN'
                else:
                    tag = 'NN'
                
                predicted.append(f"{word}/{tag}")
            out_f.write(' '.join(predicted) + '\n')

def predict_contextual(model_data, test_file, output_file):
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    index = model_data['index']
    tags = model_data['tags']
    word_tags = model_data['word_tags']
    
    with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            tokens = line.strip().split()
            words = [t.rsplit('/', 1)[0] for t in tokens]
            
            inputs = tokenizer(words, is_split_into_words=True,
                             return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            predicted_tags = []
            for word_idx in range(len(words)):
                # Get embedding for first subword token
                token_indices = [i for i, wid in enumerate(inputs.word_ids()) 
                               if wid == word_idx]
                if token_indices:
                    embedding = outputs.last_hidden_state[0, token_indices[0]].numpy().astype('float32')
                    _, I = index.search(embedding.reshape(1, -1), 1)
                    predicted_tag = tags[I[0][0]]
                else:
                    # Fallback to word-based prediction
                    word = words[word_idx]
                    if word in word_tags:
                        predicted_tag = max(word_tags[word], key=word_tags[word].get)
                    else:
                        predicted_tag = 'NN'
                
                predicted_tags.append(f"{words[word_idx]}/{predicted_tag}")
            
            out_f.write(' '.join(predicted_tags) + '\n')

def compare_results(gold_file, pred_file):
    y_true = []
    y_pred = []
    
    with open(gold_file, 'r') as f_true, open(pred_file, 'r') as f_pred:
        for true_line, pred_line in zip(f_true, f_pred):
            true_tokens = true_line.strip().split()
            pred_tokens = pred_line.strip().split()
            
            for true_token, pred_token in zip(true_tokens, pred_tokens):
                true_word, true_tag = true_token.rsplit('/', 1)
                pred_word, pred_tag = pred_token.rsplit('/', 1)
                
                if true_word == pred_word:
                    y_true.append(true_tag)
                    y_pred.append(pred_tag)
    
    precision = precision_score(y_true, y_pred, average='micro')
    print(f"Accuracy: {precision:.4f}\n")

if __name__ == "__main__":
    main()
    
# NOWORDVECTORS = 1
# STATICWORDVECTORS = 2
# CONTEXTUALWORDVECTORS = 3
# MODEL = 0
# CLASSIFIER = 1
# LABELENCODER = 2



# def main():
#     train_file = 'pos/data/ass1-tagger-train'
#     test_file = 'pos/data/ass1-tagger-dev-input'
#     test_output_file = 'pos/data/ass1-tagger-dev'

#     # # # Part 1 # # #
#     word_count = train(train_file, NOWORDVECTORS)
#     predict(test_file, test_output_file, NOWORDVECTORS, word_count, compare=True)
#     # # # Part 2 # # #
#     w2v_model, classifier, label_encoder = train(train_file, STATICWORDVECTORS)
#     predict(test_file, test_output_file, STATICWORDVECTORS, [w2v_model, classifier, label_encoder], compare=True)
#     # # # Part 3 # # #
#     # model = train(train_file, CONTEXTUALWORDVECTORS)
#     # predict(test_file, test_output_file, CONTEXTUALWORDVECTORS, model, compare=True)
    

# def train(train_file, part):
#     model = None
#     # based on part, train the model and return it
#     if part == NOWORDVECTORS:
#         model = train_word_counts(train_file)
#     elif part == STATICWORDVECTORS:
#         model = train_word2vec(train_file)
#     elif part == CONTEXTUALWORDVECTORS:
#         pass
#     else:
#         print("Invalid part number")
#     return model
    
# def predict(test_path, test_output_path, part, model, compare=False):
#     # based on part, predict the POS tags for the test file
#     output_file = f'POS_preds_{part}.txt'
#     if part == NOWORDVECTORS:
#         predict_wordcount(model, test_path , output_file)
#     elif part == STATICWORDVECTORS:
#         predict_word2vec(model[MODEL], model[CLASSIFIER], model[LABELENCODER], test_path, output_file)
#     elif part == CONTEXTUALWORDVECTORS:
#         pass
#     else:
#         print("Invalid part number")
#         exit()
#     if compare:
#         compare_results(test_output_path, output_file)
#     pass

# def print_precision(y_true, y_pred):
#     precision = precision_score(y_true, y_pred, average='micro')
#     print(f'Precision: {precision:.4f}')

# def compare_results(true_file, pred_file):
#     y_true = []
#     y_pred = []
    
#     with open(true_file, 'r') as f_true, open(pred_file, 'r') as f_pred:
#         for true_line, pred_line in zip(f_true, f_pred):
#             true_tokens = true_line.strip().split()
#             pred_tokens = pred_line.strip().split()
            
#             for true_token, pred_token in zip(true_tokens, pred_tokens):
#                 true_word, true_tag = true_token.rsplit('/', 1)
#                 pred_word, pred_tag = pred_token.rsplit('/', 1)
                
#                 if true_word == pred_word:
#                     y_true.append(true_tag)
#                     y_pred.append(pred_tag)
    
#     precision = precision_score(y_true, y_pred, average='micro')
#     print(f'Precision: {precision:.4f}')
    
# # # # Part 1 functions # # #

# def train_word_counts(train_file):
#     # create count collection
#     word_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
#     with open(train_file, 'r') as f: # opens file
#         for line in f: # get word tokens in every line
#             tokens = line.strip().split()
#             for token in tokens:
#                 word, tag = token.rsplit('/', 1)
#                 word_tag_counts[word][tag] += 1 # add count
#     return word_tag_counts

# def predict_wordcount(word_tag_counts, test_file, output_file):
#     # creates output file
#     with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
#         for line in f: # get word tokens in every line
#             tokens = line.strip().split()
#             predicted_tags = []
#             for token in tokens:
#                 word = token.rsplit('/', 1)[0]
#                 if word in word_tag_counts: # if word is in the count collection
#                     predicted_tag = max(word_tag_counts[word], key=word_tag_counts[word].get)
#                 else: # else default tag is NN
#                     predicted_tag = 'NN'
#                 predicted_tags.append(f"{word}/{predicted_tag}")
#             out_f.write(' '.join(predicted_tags) + '\n') # write to output file

# # # # Part 2 functions # # #

# def train_word2vec(train_file):
#     # load word2vec model
#     model = dl.load("glove-wiki-gigaword-200")
#     # preprocess the train file
#     X, y = get_word_tag_pairs(train_file)
#     # get word vectors for each word in the train file
#     X_vectors = [[get_word_vector(word, model) for word in sentence] for sentence in X]
#     y_flat = [tag for sentence in y for tag in sentence]
#     X_train = [vector for sentence in X_vectors for vector in sentence]
#     # Convert POS tags to numerical labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_flat)
#     # Train a classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_encoded)
#     return model, classifier, label_encoder

# def get_word_tag_pairs(file_path):
#     X, y = [], []
#     with open(file_path, 'r') as f:
#         for line in f:
#             tokens = line.strip().split()
#             words, tags = [], []
#             for token in tokens:
#                 word, tag = token.rsplit('/', 1)
#                 words.append(word)
#                 tags.append(tag)
#             X.append(words)
#             y.append(tags)
#     return X, y

# def get_word_vector(word, model):
#     try:
#         return model[word]
#     except KeyError:
#         return numpy.zeros(model.vector_size)
    
    
# def predict_word2vec(model, classifier, label_encoder, test_file, output_file):
#     with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
#         for line in f:
#             tokens = line.strip().split()
#             words = [token.rsplit('/', 1)[0] for token in tokens]
#             X_test = [get_word_vector(word, model) for word in words]
#             y_pred = classifier.predict(X_test)
#             tags_pred = label_encoder.inverse_transform(y_pred)
#             out_f.write(' '.join(f'{word}/{tag}' for word, tag in zip(words, tags_pred)) + '\n')
# # def train_word2vec(train_file):
# #     # load word2vec model
# #     model = dl.load("glove-wiki-gigaword-200")
#     # preprocess the train file
#     X, y = get_word_tag_pairs(train_file)
#     # get word vectors for each word in the train file
#     X_vectors = [[get_word_vector(word, model) for word in sentence] for sentence in X]
#     y_flat = [tag for sentence in y for tag in sentence]
#     X_train = [vector for sentence in X_vectors for vector in sentence]
#     # Convert POS tags to numerical labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_flat)
#     # Train a classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_encoded)
#     return model, classifier, label_encoder

    
# def predict_word2vec(model, classifier, label_encoder, test_file, output_file):
#     with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
#         for line in f:
#             tokens = line.strip().split()
#             words = [token.rsplit('/', 1)[0] for token in tokens]
#             X_test = [get_word_vector(word, model) for word in words]
#             y_pred = classifier.predict(X_test)
#             tags_pred = label_encoder.inverse_transform(y_pred)
#             out_f.write(' '.join(f'{word}/{tag}' for word, tag in zip(words, tags_pred)) + '\n')
            
if __name__ == "__main__":
    main()

