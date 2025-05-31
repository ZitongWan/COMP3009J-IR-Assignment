# COMP3009J Information Retrieval - Small and Large Corpus

## Notification

* All code runs within the command line and based on testing, all code can run normally.

* If you are using **Conda**, please make sure to activate the corresponding Conda environment,eg: `conda activate xxx`

* If using a **local root environment**, please ensure that Python can be found correctly.

* For the operation of small and large corpus, please refer
  to [small corpus](./comp3009j-corpus-small/README.md), [large corpus](./comp3009j-corpus-large/README.md).

## Small Corpus Directory structure(Same as large Corpus)

```
COMP3009J-corpus-small(large)/
├──documents/
│  ├── ...
├──files/
│  ├── porter.py
│  ├── qrels.txt
│  ├── queries.txt
│  ├── sample_ourput.txt
│  ├── stopwords.txt

├──index-small(large)-corpus.py
├──query-small(large)-corpus.py
├──evaluate-small(large)-corpus.py
├──Number-small(large).index.json (After running relevant code)
├──Number-small(large).results (After running relevant code)
```

## Efficiency Improvement

Considering a large corpus, **stemming word** is a significant challenge. The normal operation is to use steamer.stem()
one by one, which greatly increases duplicate calculations and the IO overhead. However, storing the results after each
steam in a dictionary and directly retrieving the second and subsequent steam from it can greatly optimize the speed,
similar to **caching**.

After using this method:

For small corpus: the running time of extracting documents decreased from around 1.1s to around 0.6~0.7s.

For large corpus: the running time of extracting documents decreased from around 120s to around 20~25s.

## Parameter Selection

### Grid Search for parameters of k,b in BM25

**Note: (small/large) results.txt includes top 30 results for each query in automatic mode ---
query_small/large_corpus.py**

The following is a list of parameters for grid search (both for small and large corpus):

- k_list = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

- b_list = [0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.8, 0.9]

Both of them have undergone over 200 experiments to screen based on the evaluation results as an indicator.

The performance of each combination in the top 10 is effective, but if we have to choose one, the one with the best
average performance is a good choice.

For small corpus: **k = 3.0, b =0.8**

For large corpus: **k = 3.5, b =0.75**

### BM25 Hyperparameter Tuning Results (Top 10)

#### Small Corpus:

| Rank | k    | b    | Precision | Recall | R-Precision | P@15   | NDCG@15 | Map    | avgScore |
|------|------|------|-----------|--------|-------------|--------|---------|--------|----------|
| 1    | 3.0  | 0.8  | 0.1527    | 0.6202 | 0.3999      | 0.2424 | 0.4148  | 0.4046 | 0.3724   |
| 2    | 3.2  | 0.8  | 0.1520    | 0.6180 | 0.4002      | 0.2433 | 0.4156  | 0.4046 | 0.3723   |
| 3    | 3.5  | 0.75 | 0.1521    | 0.6152 | 0.4037      | 0.2439 | 0.4153  | 0.4035 | 0.3723   |
| 4    | 4.0  | 0.75 | 0.1519    | 0.6152 | 0.4024      | 0.2453 | 0.4156  | 0.4031 | 0.3723   |
| 5    | 3.5  | 0.8  | 0.1520    | 0.6176 | 0.4008      | 0.2441 | 0.4154  | 0.4032 | 0.3722   |
| 6    | 4.5  | 0.75 | 0.1514    | 0.6150 | 0.4028      | 0.2450 | 0.4161  | 0.4016 | 0.3720   |
| 7    | 3.2  | 0.75 | 0.1520    | 0.6161 | 0.4003      | 0.2439 | 0.4157  | 0.4029 | 0.3718   |
| 8    | 4.2  | 0.75 | 0.1513    | 0.6134 | 0.4022      | 0.2453 | 0.4160  | 0.4024 | 0.3718   |
| 9    | 4.15 | 0.75 | 0.1516    | 0.6143 | 0.4024      | 0.2444 | 0.4153  | 0.4026 | 0.3717   |
| 10   | 4.3  | 0.75 | 0.1513    | 0.6143 | 0.4018      | 0.2450 | 0.4155  | 0.4021 | 0.3717   |

##### Large Corpus:

| Rank | k   | b    | Precision | Recall | R-Precision | P@15   | NDCG@15 | Map    | Bpref  | Avg Score |
|------|-----|------|-----------|--------|-------------|--------|---------|--------|--------|-----------|
| 1    | 3.5 | 0.75 | 0.4827    | 0.7708 | 0.5200      | 0.5506 | 0.5786  | 0.4970 | 0.5020 | 0.5574    |
| 2    | 3.2 | 0.75 | 0.4819    | 0.7695 | 0.5236      | 0.5490 | 0.5767  | 0.4968 | 0.5023 | 0.5571    |
| 3    | 4.5 | 0.75 | 0.4827    | 0.7685 | 0.5201      | 0.5498 | 0.5758  | 0.4978 | 0.4995 | 0.5563    |
| 4    | 3.0 | 0.75 | 0.4802    | 0.7674 | 0.5219      | 0.5481 | 0.5760  | 0.4955 | 0.5023 | 0.5559    |
| 5    | 4.0 | 0.9  | 0.4856    | 0.7793 | 0.5202      | 0.5399 | 0.5721  | 0.4932 | 0.4996 | 0.5557    |
| 6    | 3.0 | 0.8  | 0.4807    | 0.7654 | 0.5232      | 0.5473 | 0.5754  | 0.4951 | 0.5023 | 0.5556    |
| 7    | 3.2 | 0.8  | 0.4807    | 0.7658 | 0.5234      | 0.5473 | 0.5755  | 0.4950 | 0.5011 | 0.5555    |
| 8    | 3.5 | 0.8  | 0.4811    | 0.7670 | 0.5248      | 0.5457 | 0.5719  | 0.4954 | 0.5024 | 0.5555    |
| 9    | 2.5 | 0.75 | 0.4774    | 0.7646 | 0.5221      | 0.5506 | 0.5775  | 0.4945 | 0.5015 | 0.5554    |
| 10   | 4.5 | 0.8  | 0.4811    | 0.7681 | 0.5240      | 0.5440 | 0.5732  | 0.4958 | 0.5006 | 0.5553    |

Comparing the experimental results of the two, the difference between the small corpus is slightly smaller, with a
difference of only (0.0007) between the first and tenth places, while the large corpus has (0.0021).

The value of k fluctuates relatively large (with a difference about 1), while the value of b is relatively fixed, at
0.75 or 0.8.
