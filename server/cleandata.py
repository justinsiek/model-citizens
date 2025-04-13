import gc
import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

class Config:
    seed = 42
    n_splits = 10
    sample_size = None  

def process(input_str):
    if isinstance(input_str, str):
        stripped_str = input_str.strip('[]')
        sentences = [s.strip('"') for s in stripped_str.split('","')]
        return ' '.join(sentences)
    return input_str

class Preprocessor:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2'):
        print(f"Loading SBERT model: {sbert_model_name}...")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        print("SBERT model loaded.")

    def cosine_sim(self, text1: str, text2: str):
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            vectorizer.fit([text1, text2])
            output = vectorizer.transform([text1, text2]).toarray()
            cos_sim = cosine_similarity(output)
            return cos_sim[0][1]
        except:
            return np.nan

    def jaccard_sim(self, text1: str, text2: str):
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) > 0 else 0
    
    def count_new_lines(self, text: str) -> int:
        return text.count('\\n')
    
    def count_quotes(self, text: str) -> int:
        single_quote_pattern = r"'(.*?)'"
        double_quote_pattern = r'"(.*?)"'
        single_quotes = re.findall(single_quote_pattern, text)
        double_quotes = re.findall(double_quote_pattern, text)
        return len(single_quotes) + len(double_quotes)

    def tokenize(self, text: str):
        return nltk.word_tokenize(text.lower())

    def generate_ngrams(self, text: str, n: int):
        tokens = self.tokenize(text)
        return list(ngrams(tokens, n))

    def count_ngram_overlaps(self, text1: str, text2: str, n: int) -> int:
        try:
            ngrams1 = self.generate_ngrams(text1, n)
            ngrams2 = self.generate_ngrams(text2, n)
            counter1 = Counter(ngrams1)
            counter2 = Counter(ngrams2)
            overlap = counter1 & counter2
            overlap_count = sum(overlap.values())
            return overlap_count
        except:
            return 0
        
    def sbert_sim(self, text1: str, text2: str):
        """Calculates cosine similarity between SBERT embeddings."""
        try:
            embeddings = self.sbert_model.encode([text1, text2], convert_to_tensor=True)
            # Calculate cosine similarity (using PyTorch's cosine_similarity)
            # Note: The sentence_transformers library includes utility functions for this,
            # but using the core formula directly avoids extra imports if needed elsewhere.
            # Ensure embeddings are on CPU for numpy conversion if needed,
            # or use torch functions directly.
            emb1 = embeddings[0].unsqueeze(0) # Add batch dimension
            emb2 = embeddings[1].unsqueeze(0) # Add batch dimension
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            return cos_sim
        except Exception as e:
            print(f"Error calculating SBERT similarity: {e}")
            return np.nan

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Extracting features...")
        
        data["respa_respb_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 1), axis=1)
        data["respa_respb_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 2), axis=1)
        data["respa_respb_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 3), axis=1)

        data["respa_prompt_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 1), axis=1)
        data["respa_prompt_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 2), axis=1)
        data["respa_prompt_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 3), axis=1)

        data["respb_prompt_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 1), axis=1)
        data["respb_prompt_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 2), axis=1)
        data["respb_prompt_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 3), axis=1)
        
        data["respa_len"] = data["response_a"].apply(lambda x: len(self.tokenize(x)))
        data["respb_len"] = data["response_b"].apply(lambda x: len(self.tokenize(x)))
        data["prompt_len"] = data["prompt"].apply(lambda x: len(self.tokenize(x)))
        
        data["respa_new_lines"] = data["response_a"].apply(lambda x: self.count_new_lines(x))
        data["respb_new_lines"] = data["response_b"].apply(lambda x: self.count_new_lines(x))
        data["prompt_new_lines"] = data["prompt"].apply(lambda x: self.count_new_lines(x))
        
        data["respa_prompt_len_ratio"] = data["respa_len"] / data["prompt_len"]
        data["respb_prompt_len_ratio"] = data["respb_len"] / data["prompt_len"]
        data["respa_respb_len_ratio"] = data["respa_len"] / data["respb_len"]
        
        data["respa_respb_len_diff"] = data["respa_len"] - data["respb_len"]
        data["respa_prompt_len_diff"] = data["respa_len"] - data["prompt_len"]
        data["respb_prompt_len_diff"] = data["respb_len"] - data["prompt_len"]
        
        # Handle division by zero
        data["respa_prompt_overlap_unigram_len_ratio"] = data.apply(
            lambda x: x["respa_prompt_overlap_unigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)
        data["respa_prompt_overlap_bigram_len_ratio"] = data.apply(
            lambda x: x["respa_prompt_overlap_bigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)
        data["respa_prompt_overlap_trigram_len_ratio"] = data.apply(
            lambda x: x["respa_prompt_overlap_trigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)

        data["respb_prompt_overlap_unigram_len_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_unigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)
        data["respb_prompt_overlap_bigram_len_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_bigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)
        data["respb_prompt_overlap_trigram_len_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_trigram"] / x["prompt_len"] if x["prompt_len"] > 0 else 0, axis=1)
        
        data["overlap_unigram_diff"] = data["respa_prompt_overlap_unigram"] - data["respb_prompt_overlap_unigram"]
        data["overlap_bigram_diff"] = data["respa_prompt_overlap_bigram"] - data["respb_prompt_overlap_bigram"]
        data["overlap_trigram_diff"] = data["respa_prompt_overlap_trigram"] - data["respb_prompt_overlap_trigram"]
        
        # Handle division by zero
        data["overlap_unigram_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_unigram"] / x["respa_prompt_overlap_unigram"] 
            if x["respa_prompt_overlap_unigram"] > 0 else 0, axis=1)
        data["overlap_bigram_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_bigram"] / x["respa_prompt_overlap_bigram"] 
            if x["respa_prompt_overlap_bigram"] > 0 else 0, axis=1)
        data["overlap_trigram_ratio"] = data.apply(
            lambda x: x["respb_prompt_overlap_trigram"] / x["respa_prompt_overlap_trigram"] 
            if x["respa_prompt_overlap_trigram"] > 0 else 0, axis=1)
        
        data["respa_quotes"] = data["response_a"].apply(lambda x: self.count_quotes(x))
        data["respb_quotes"] = data["response_b"].apply(lambda x: self.count_quotes(x))
        data["prompt_quotes"] = data["prompt"].apply(lambda x: self.count_quotes(x))
        
        print("Computing similarity metrics...")
        data["respa_respb_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_a"], x["response_b"]), axis=1)
        data["respa_respb_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_a"], x["response_b"]), axis=1)
        
        data["respa_prompt_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_a"], x["prompt"]), axis=1)
        data["respa_prompt_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_a"], x["prompt"]), axis=1)
        
        data["respb_prompt_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_b"], x["prompt"]), axis=1)
        data["respb_prompt_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_b"], x["prompt"]), axis=1)
        
        data["jaccard_sim_diff"] = data["respa_prompt_jaccard_sim"] - data["respb_prompt_jaccard_sim"]
        data["jaccard_sim_ratio"] = data.apply(
            lambda x: x["respb_prompt_jaccard_sim"] / x["respa_prompt_jaccard_sim"] 
            if x["respa_prompt_jaccard_sim"] > 0 else 0, axis=1)
        
        print("Computing SBERT similarity features...")
        tqdm.pandas(desc="SBERT Similarities")
        
        # Calculate SBERT similarities with progress bars
        data["respa_respb_sbert_sim"] = data.progress_apply(lambda x: self.sbert_sim(x["response_a"], x["response_b"]), axis=1)
        data["respa_prompt_sbert_sim"] = data.progress_apply(lambda x: self.sbert_sim(x["response_a"], x["prompt"]), axis=1)
        data["respb_prompt_sbert_sim"] = data.progress_apply(lambda x: self.sbert_sim(x["response_b"], x["prompt"]), axis=1)

        # Derived SBERT features (optional, but might be useful)
        data["sbert_sim_diff"] = data["respa_prompt_sbert_sim"] - data["respb_prompt_sbert_sim"]
        data["sbert_sim_ratio"] = data.apply(
            lambda x: x["respb_prompt_sbert_sim"] / x["respa_prompt_sbert_sim"]
            if x["respa_prompt_sbert_sim"] != 0 else np.nan, axis=1 # Avoid division by zero, use nan
        )
        
        return data

def prepare_data():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    
    # Keep only relevant columns and filter out ties
    filtered_df = train_df[train_df['winner_tie'] == 0][['prompt', 'response_a', 'response_b', 'winner_model_b']]
    
    # Sample a smaller dataset for faster processing if sample_size is set
    original_size = len(filtered_df)
    if Config.sample_size is not None and original_size > Config.sample_size:
        filtered_df = filtered_df.sample(Config.sample_size, random_state=Config.seed)
        print(f"Sampled {Config.sample_size} examples from {original_size} (no ties)")
    else:
        print(f"Using full dataset with {len(filtered_df)} examples (no ties)")
    
    # Check we have both binary classes
    class_counts = filtered_df['winner_model_b'].value_counts()
    print(f"Class distribution: model A wins: {class_counts.get(0, 0)}, model B wins: {class_counts.get(1, 0)}")
    
    # Apply text processing
    filtered_df["prompt"] = filtered_df["prompt"].apply(process)
    filtered_df["response_a"] = filtered_df["response_a"].apply(process)
    filtered_df["response_b"] = filtered_df["response_b"].apply(process)
    
    # Add target column (binary: 0 for model A wins, 1 for model B wins)
    filtered_df["target"] = filtered_df["winner_model_b"]
    
    # Extract features
    preprocessor = Preprocessor()
    processed_df = preprocessor.run(filtered_df)
    
    # Replace infinities with NaN
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    
    # Save processed data
    processed_df.to_csv('processed_data.csv', index=False)
    print("Data processing complete. Saved to processed_data.csv")
    
    return processed_df

if __name__ == "__main__":
    prepare_data()