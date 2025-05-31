"""
StudentID: 22207296
StudentName: Zitong Wan

The strategy for saving results here is to return the top 30 documents with the highest results.
"""

import os
import json
import re
import time
import argparse
from files import porter
from collections import defaultdict


class QueryProcessor:
    def __init__(self, index_file, stopwords_path):
        self.index = self.load_index_file(index_file)
        self.stopwords = self.load_stopwords(stopwords_path)
        # Match English words (including abbreviations such as I'm) or numbers (including 3.2b format), ignoring capitalization
        self.pattern = re.compile(r"[a-z\']+|(?:\d+\.\db+|\d+)", flags=re.I)
        self.stem_cache = {}  # stem cache, similar to caching, improving processing speed
        self.results = {}
        self.stemmer = porter.PorterStemmer()

    def load_index_file(self, index_file):
        """Load the index file --- json file"""
        print("Loading the Small Index file...")
        start_time = time.time()
        with open(index_file, 'r', encoding="utf8") as f:
            index = json.load(f)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Loading cost {duration:.4f}s")

        return index

    def load_stopwords(self, stopwords_path):
        """Load stopwords from stopwords path"""
        with open(stopwords_path, 'r', encoding='utf8') as f:
            stopwords = {word.strip() for word in f.readlines()}

        return stopwords

    def remove_stopwords(self, query):
        """Remove stopword from a document"""
        return [term for term in query if term not in self.stopwords]  # Inversion screen

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

    def extract_query(self, query):
        """Extract the query as a token"""
        tokens = self.pattern.findall(query.lower())
        filtered_tokens = self.remove_stopwords(tokens)
        stemmed_tokens = self.stem_words(filtered_tokens)

        return stemmed_tokens

    def query_results(self, query):
        """Return the result after querying"""
        tokens = self.extract_query(query)
        results = defaultdict(float)
        for doc_id, doc_tokens in self.index.items():
            query_score = sum(doc_tokens.get(token, 0) for token in tokens)
            if query_score != 0:
                results[doc_id] = query_score

        return results

    def process_query(self, query):
        """Perform the query and return sorted results."""
        self.extract_query(query)
        self.results = self.query_results(query)

        return self.results


class Mode:
    def __init__(self, query_processor, queries_file, output_file):
        self.query_processor = query_processor
        self.queries_file = queries_file
        self.output_file = output_file

    def automatic_mode(self):
        """
        Read queries from file and write results to the same directory as the script.
        Here are the top 30 documents of the query results returned.
        """
        with open(self.queries_file, 'r', encoding='utf8') as q, open(self.output_file, 'w', encoding='utf8') as o:
            # Calculate all query time.
            total_time = 0
            for line in q:
                query_id, query = line.strip().split(' ', 1)
                start_time = time.time()
                query_results = sorted(self.query_processor.process_query(query).items(), key=lambda x: x[1],
                                       reverse=True)[:30]
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                for rank, (doc_id, score) in enumerate(query_results, start=1):
                    o.write(f"{query_id} {doc_id} {rank} {score:.4f}\n")
            print(f"Queries completed in {total_time:.4f} seconds and file has been saved to {self.output_file}...")

    def interactive_mode(self):
        """Input queries and print results."""
        while True:
            query = input("Please input query:")
            print()
            if query.lower() != "exit":
                # Calculate the query time.
                start_time = time.time()
                query_results = sorted(self.query_processor.process_query(query).items(), key=lambda x: x[1],
                                       reverse=True)[:15]
                end_time = time.time()
                duration = end_time - start_time
                print(f"Query completed in {duration:.4f} seconds...")

                if query_results:
                    print(f"Query: {query}")
                    print(f"{'Rank':<8}{'Doc ID':<12}{'Score':<14}{'StudentId':<8}")
                    for rank, (doc_id, score) in enumerate(query_results, start=1):
                        print(f"{rank:<8}{doc_id:<12}{score:<14.6f}{'22207296':<8}")
                    print()
                else:
                    print(f"No results found for search query: {query}...")
            else:
                print("Bye bye~")
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Query_small_corpus")
    parser.add_argument('-m', '--mode', type=str, choices=['interactive', 'automatic'], required=True,
                        help="Mode of query...")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the small corpus directory...")

    args = parser.parse_args()

    index_path = os.path.join(args.path, "22207296-small.index.json")
    stopwords_path = os.path.join(args.path, "files", "stopwords.txt")

    if not os.path.exists(index_path):
        print("Index file not found.")
    elif not os.path.exists(stopwords_path):
        print("Stopwords file not found.")
    else:
        query_processor = QueryProcessor(index_path, stopwords_path)

        queries_file = os.path.join(args.path, "files", "queries.txt")
        output_file = os.path.join(os.getcwd(), "22207296-small.results")

        mode = Mode(query_processor, queries_file, output_file)

        if args.mode == 'interactive':
            mode.interactive_mode()
        elif args.mode == 'automatic':
            mode.automatic_mode()
        else:
            print("Invalid mode.")
