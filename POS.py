import collections
from sklearn.metrics import precision_score
import numpy
import gensim.downloader as dl
from gensim.models import Word2Vec

NOWORDVECTORS = 1
STATICWORDVECTORS = 2
CONTEXTUALWORDVECTORS = 3


def main():
    train_file = 'pos/data/ass1-tagger-train'
    test_file = 'pos/data/ass1-tagger-dev-input'
    test_output_file = 'pos/data/ass1-tagger-dev'

    # # # Part 1 # # #
    word_count = train(train_file, NOWORDVECTORS)
    predict(test_file, test_output_file, NOWORDVECTORS, word_count, compare=True)
    # # # Part 2 # # #
    w2v_model = train(train_file, STATICWORDVECTORS)
    predict(test_file, test_output_file, STATICWORDVECTORS, [w2v_model, word_count], compare=True)
    # # # Part 3 # # #
    # model = train(train_file, CONTEXTUALWORDVECTORS)
    #predict(test_file, test_output_file, CONTEXTUALWORDVECTORS, model, compare=True)
    

def train(train_file, part):
    model = None
    # based on part, train the model and return it
    if part == NOWORDVECTORS:
        model = train_word_counts(train_file)
    elif part == STATICWORDVECTORS:
        model = train_word2vec(train_file)
    elif part == CONTEXTUALWORDVECTORS:
        pass
    else:
        print("Invalid part number")
    return model
    
def predict(test_path, test_output_path, part, model, compare=False):
    # based on part, predict the POS tags for the test file
    output_file = f'POS_preds_{part}.txt'
    if part == NOWORDVECTORS:
        predict_wordcount(model, test_path , output_file)
    elif part == STATICWORDVECTORS:
        predict_word2vec(model[0], model[1], test_path, output_file)
    elif part == CONTEXTUALWORDVECTORS:
        pass
    else:
        print("Invalid part number")
        exit()
    if compare:
        compare_results(test_output_path, output_file)
    pass

def print_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro')
    print(f'Precision: {precision:.4f}')

def compare_results(true_file, pred_file):
    y_true = []
    y_pred = []
    
    with open(true_file, 'r') as f_true, open(pred_file, 'r') as f_pred:
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
    print(f'Precision: {precision:.4f}')
    
# # # Part 1 functions # # #

def train_word_counts(train_file):
    # create count collection
    word_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(train_file, 'r') as f: # opens file
        for line in f: # get word tokens in every line
            tokens = line.strip().split()
            for token in tokens:
                word, tag = token.rsplit('/', 1)
                word_tag_counts[word][tag] += 1 # add count
    return word_tag_counts

def predict_wordcount(word_tag_counts, test_file, output_file):
    # creates output file
    with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f: # get word tokens in every line
            tokens = line.strip().split()
            predicted_tags = []
            for token in tokens:
                word = token.rsplit('/', 1)[0]
                if word in word_tag_counts: # if word is in the count collection
                    predicted_tag = max(word_tag_counts[word], key=word_tag_counts[word].get)
                else: # else default tag is NN
                    predicted_tag = 'NN'
                predicted_tags.append(f"{word}/{predicted_tag}")
            out_f.write(' '.join(predicted_tags) + '\n') # write to output file

# # # Part 2 functions # # #

def train_word2vec(train_file):
    # load word2vec model
    model = dl.load('word2vec-google-news-300')
    

def predict_word2vec(model, word_tag_counts, test_file, output_file):
    # creates output file
    with open(test_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f: # get word tokens in every line
            tokens = line.strip().split()
            predicted_tags = []
            for token in tokens:
                word = token.rsplit('/', 1)[0]
                if word in word_tag_counts: # if word is in the count collection
                    predicted_tag = max(word_tag_counts[word], key=word_tag_counts[word].get)
                else: # else find closest word in word2vec model and get its tag
                    similar_words = model.wv.most_similar(word)
                    for similar_word in similar_words:
                        print(similar_word)
                        if similar_word[0] in word_tag_counts:
                            predicted_tag = max(word_tag_counts[similar_word[0]], key=word_tag_counts[similar_word[0]].get)
                            break
                    else:
                        predicted_tag = 'NN'
                predicted_tags.append(f"{word}/{predicted_tag}")
            out_f.write(' '.join(predicted_tags) + '\n') # write to output file
            
            
if __name__ == "__main__":
    main()

