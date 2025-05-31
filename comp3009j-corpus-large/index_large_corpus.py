"""
StudentID: 22207296
StudentName: Zitong Wan

You can read the overall README.file to view the necessary information, such as directory structure, grid search for k, and b parameters.
- [README.md]

The parameter search range for k and b:
k_list = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
b_list = [0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.8, 0.9]

Ultimately, for large corpus, the best choice of k is 3.5, b is 0.75 (Based on the evaluation scores, you can read the relevant README file).

You can read the comp3009j-corpus-large/README.md file to see more necessary information about the commands for running.
- [comp3009j-corpus-large/README.md]
"""

import json
import math
import os
import re
import time
import argparse

from files import porter
from collections import defaultdict


class DocumentProcessor:
    def __init__(self, doc_path, stopwords_path):
        self.stem_cache = {}  # stem cache, similar to caching, improving processing speed
        self.stemmer = porter.PorterStemmer()
        self.stopwords = self.load_stopwords(stopwords_path)
        # Match English words (including abbreviations such as I'm) or numbers (including 3.2b format), ignoring capitalization
        self.pattern = re.compile(r"[a-z\']+|(?:\d+\.\db+|\d+)", flags=re.I)
        self.docs, self.docs_num, self.docs_len, self.avg_doc_len = self.extract_documents(doc_path)  # Preloading data

    def load_stopwords(self, stopwords_path):
        """Load stopwords from stopwords path"""
        with open(stopwords_path, 'r') as f:
            stopwords = {word.strip() for word in f.readlines()}
        return stopwords

    def remove_stopwords(self, document):
        """Remove stopword from a document"""
        return [term for term in document if term not in self.stopwords]  # Inversion screen

    def stem_words(self, document):
        """
        Steam cache dynamically increases with document processing
        If a term is inside, use it directly. If not, add its restored stem to the stem cache
        Similar to caching
        """

        stemmed_terms = []
        for term in document:
            if term in self.stem_cache:
                stemmed_terms.append(self.stem_cache[term])
            else:
                stemmed_term = self.stemmer.stem(term)
                self.stem_cache[term] = stemmed_term
                stemmed_terms.append(stemmed_term)

        return stemmed_terms

    def extract_documents(self, doc_path):
        """Execute the complete document processing workflow: reading, removing stopwords, and stemming"""
        print("Extracting all documents......")
        docs = {}
        docs_num = 0
        docs_len = 0

        # Calculate running time
        start_time = time.time()
        for root, dirs, files in os.walk(doc_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding="utf8") as f:
                        text = f.read().lower()

                        tokens = self.pattern.findall(text)  # Extract documents based on the strategy of self.pattern

                        filtered_tokens = self.remove_stopwords(tokens)  # Delete stop words
                        stemmed_tokens = self.stem_words(filtered_tokens)  # Stem words

                        docs_num += 1  # Count the number of documents
                        docs[filename] = stemmed_tokens  # Add the document to the dictionary
                        docs_len += len(stemmed_tokens)  # Calculate document length

                        #  Statistical progress
                        if docs_num % 100 == 0:
                            print(f"Processed {docs_num} documents...")

                except Exception as e:
                    print(f"Error processing file: {file_path}, error: {e}")

        print(f"Process over and total documents: {docs_num}")

        avg_doc_len = docs_len / docs_num if docs_num > 0 else 0  # Calculate the average doc length to facilitate the subsequent calculation of BM25

        end_time = time.time()
        duration = end_time - start_time
        print(f"Extract documents costs {duration:.4f} s...")
        print()

        return docs, docs_num, docs_num, avg_doc_len


class BM25LargeIndex:
    def __init__(self, document_processor, k=3.5, b=0.75):
        """
        Here, I did not follow the instructions given in the courseware where k=1 and b=0.75, but rather k=3.5 and b=0.75.
        I used the grid search strategy and tried over 200 combinations and debugging large corpus, which took me almost three hours.

        The criteria for evaluation are the mean of precision, recall, r-precision, p@15, map, NDCG@15 and Bpref.
        The experimental results of the run have a small difference in mean, with the smallest reaching 0.0002, and there is not much difference in each indicator.
        So I ultimately chose the top ranked (judged by avg Score) combination as the choice for k, b.
        Based on the scores of different combinations of k and b in the evaluation, k=3.5 and b=0.75 (Rank 1) were ultimately chosen.
        It's not that other combinations are bad, other combinations perform better than the first place in certain indicators.
        But when having to choose a value of k or b, choosing the mean(avg Score) as the first option is still a good choice.

        According to my search strategy, for large corpus, the range of k is (2.5-4.5), while b is relatively fixed at 0.75/0.8.
        The average score with k=3.5 and b=0.75 is the best, but the difference between it and other combinations is very small, such as only 0.0003.
        For example, the second rank with k=3.2 and b=0.75 has higher r-precision and bpref scores than the first ranked,but
            lower scores in all other aspects.Their final average score is lower than the first rank's (k=3.5, b=0.75) score of 0.0003.

        Regarding how to search for k and b in large corpus, as well as the top 10 results, you can find them in the README.md file.
        """
        self.processor = document_processor
        self.k = k
        self.b = b
        self.tf = {}
        self.idf = self.calculate_idf()  # Preloading IDF, BM25_Scores
        self.bm25_scores = self.calculate_bm25_scores()
        self.json_path = os.path.join(os.getcwd(), f"22207296-large.index.json")

    def calculate_idf(self):
        """Compute the IDF, idf_i = log2( (N-n_i+0.5) / (n_i+0.5) + 1) : N is the total number of documents, n_i is the number of documents containing term i"""
        docs = self.processor.docs
        # Here, defaultdict can automatically initialize non-existent keys, with a default value of 0 (return value of int())
        # idf: {term1: idf1, term2: idf2, ...}
        idf = defaultdict(int)
        for doc_terms in docs.values():
            for term in set(doc_terms):
                idf[term] += 1

        idf = {term: math.log2(1 + (self.processor.docs_num - count + 0.5) / (count + 0.5)) for term, count in
               idf.items()}

        return idf

    def calculate_tf(self, document):
        """Compute the TF of each term, tf_i = (count_i * (k + 1)) / (count_i + k * (1 - b + b * len(document) / avg_doc_len))"""
        tf = defaultdict(float)
        for token in document:
            tf[token] += 1

        # tf: {doc1: {term1: tf1, term2: tf1...}, doc2: {term1: tf1, term2: tf2...}, ...}
        tf = {token: (count * (1 + self.k)) / (
                count + self.k * ((1 - self.b) + self.b * len(document) / self.processor.avg_doc_len)) for
              token, count in tf.items()}

        return tf

    def calculate_bm25_score(self, document):
        """Compute the BM25 score of each document in a document"""
        scores = defaultdict(float)
        doc_tf = self.calculate_tf(document)
        for term, tf in doc_tf.items():
            idf = self.idf[term]
            scores[term] = idf * tf
        return scores

    def calculate_bm25_scores(self):
        """Compute the BM25 score of each document in all documents"""
        print("Calculate BM25 scores...")
        start_time = time.time()
        scores = {}
        # scores: {doc1: {term1: score1, term2: score2,...}, doc2: {term1: score1, term2: score2,...}, ...}
        docs = self.processor.docs
        for i, doc in enumerate(docs, start=1):
            scores[doc] = self.calculate_bm25_score(docs[doc])
            print(f"Calculating BM25 for document {i}...") if i % 1000 == 0 else None
        end_time = time.time()
        duration = end_time - start_time

        print(f"Calculation BM25 completed, cost {duration:.4f} s")
        print()
        return scores

    def save_scores_to_file(self):
        """Export BM25 scores to a Human-readable --- JSON file"""
        print("Saving BM25 scores...")
        if self.bm25_scores.keys():
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.bm25_scores, f, ensure_ascii=False, indent=4)
            print(f"📁 File has saved to: {self.json_path}")
        else:
            print("⚠️ No BM25 score can be exported")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Index_large_corpus")
    parser.add_argument('-p', "--path", type=str, required=True, help="Path to the large corpus directory...")

    args = parser.parse_args()

    documents_path = os.path.join(args.path, "documents")
    stopwords_path = os.path.join(args.path, "files", "stopwords.txt")

    if not os.path.exists(documents_path):
        print(f"⚠️ Unable to find document path: {documents_path}")
    elif not os.path.exists(stopwords_path):
        print(f"⚠️ Cannot find stopwords path: {stopwords_path}")
    else:
        document_processor = DocumentProcessor(documents_path, stopwords_path)
        large_bm25_index = BM25LargeIndex(document_processor)
        large_bm25_index.save_scores_to_file()
        print(f"Index has been exported to 22207296-large.index.json")
