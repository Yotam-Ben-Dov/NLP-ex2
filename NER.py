import sys
import os
import collections
import numpy as np
import gensim.downloader as dl
from transformers import RobertaTokenizerFast, RobertaModel
import faiss
import torch

class NERTagger:
    def __init__(self):
        self.glove = None
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.train_data = collections.defaultdict(list)
        
    def load_data(self, train_file):
        """Load training data and build frequency tables"""
        word_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        with open(train_file) as f:
            for line in f:
                for token in line.strip().split():
                    word, tag = token.rsplit('/', 1)
                    word_tag_counts[word][tag] += 1
        self.train_data['word_tag_counts'] = word_tag_counts

    def train_static(self):
        """Prepare static word vectors model"""
        self.glove = dl.load("glove-wiki-gigaword-300")
        self._build_faiss_index()

    def train_contextual(self):
        """Prepare contextual embeddings model"""
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self._build_contextual_index()

    def _build_faiss_index(self):
        """Create FAISS index for static vectors"""
        train_vectors = []
        train_tags = []
        for word in self.train_data['word_tag_counts']:
            if word in self.glove:
                train_vectors.append(self.glove[word])
                train_tags.append(max(self.train_data['word_tag_counts'][word], 
                                    key=self.train_data['word_tag_counts'][word].get))
        
        self.train_data['static_index'] = faiss.IndexFlatL2(300)
        self.train_data['static_index'].add(np.array(train_vectors).astype('float32'))
        self.train_data['static_tags'] = train_tags

    def _build_contextual_index(self):
        """Create FAISS index for contextual vectors"""
        all_embeddings = []
        all_tags = []
        
        for word in self.train_data['word_tag_counts']:
            inputs = self.roberta_tokenizer(word, return_tensors='pt')
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
            embedding = outputs.last_hidden_state[0, 0].numpy()  # Use [CLS] token
            all_embeddings.append(embedding)
            all_tags.append(max(self.train_data['word_tag_counts'][word], 
                              key=self.train_data['word_tag_counts'][word].get))
        
        self.train_data['contextual_index'] = faiss.IndexFlatL2(768)
        self.train_data['contextual_index'].add(np.array(all_embeddings).astype('float32'))
        self.train_data['contextual_tags'] = all_tags

    def predict(self, test_file, output_file, method='baseline'):
        """Generate predictions using specified method"""
        with open(test_file) as fin, open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                predictions = []
                
                for token in tokens:
                    word = token.rsplit('/', 1)[0]
                    predictions.append(self._predict_word(word, method))
                
                fout.write(' '.join([f"{word}/{tag}" for word, tag in zip(tokens, predictions)]) + '\n')

    def _predict_word(self, word, method):
        """Predict tag for a single word"""
        # Try exact match first
        if word in self.train_data['word_tag_counts']:
            return max(self.train_data['word_tag_counts'][word], 
                      key=self.train_data['word_tag_counts'][word].get)
        
        # Handle unknown words
        if method == 'baseline':
            return 'O'
        
        if method == 'static' and self.glove is not None:
            return self._static_unknown(word)
        
        if method == 'contextual' and self.roberta_model is not None:
            return self._contextual_unknown(word)
        
        return 'O'

    def _static_unknown(self, word):
        """Handle unknown words with static vectors"""
        if word in self.glove:
            vec = self.glove[word].reshape(1, -1).astype('float32')
            _, I = self.train_data['static_index'].search(vec, 1)
            return self.train_data['static_tags'][I[0][0]]
        return 'O'

    def _contextual_unknown(self, word):
        """Handle unknown words with contextual vectors"""
        inputs = self.roberta_tokenizer(word, return_tensors='pt')
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        embedding = outputs.last_hidden_state[0, 0].numpy().astype('float32')
        _, I = self.train_data['contextual_index'].search(embedding.reshape(1, -1), 1)
        return self.train_data['contextual_tags'][I[0][0]]

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python ner_tagger.py <train_file> <test_file> <gold_file> <method>")
        print("Methods: baseline, static, contextual")
        sys.exit(1)

    train_file, test_file, gold_file, method = sys.argv[1:]
    
    tagger = NERTagger()
    
    # Train selected model
    tagger.load_data(train_file)
    if method == 'static':
        tagger.train_static()
    elif method == 'contextual':
        tagger.train_contextual()

    # Generate predictions
    output_file = f"NER_preds_{method}.txt"
    tagger.predict(test_file, output_file, method)

    # Evaluate
    os.system(f"python ner_eval.py {gold_file} {output_file}")