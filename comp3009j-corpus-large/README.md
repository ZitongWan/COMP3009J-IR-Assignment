# COMP3009J Information Retrieval

## Commands

**It is assumed here that you are in the [comp3009j-corpus-large](../comp3009j-corpus-large) folder.**

Similar to this effect in the terminal,such as: **E:\\COMP3009J\\comp3009-corpus-large>**

You can see the project structure from the overall [README](../README.md) file.

* All code runs within the command line and based on testing, all code can run normally.

* If you are using Conda, please make sure to activate the corresponding Conda environment,eg: 'conda activate xxx'

* If using a local root environment, please ensure that Python can be found correctly.

**index_large_corpus.py:**

```cmd
python .\index_large_corpus.py -p .
```

**query_large_corpus.py:**

```cmd
python .\query_large_corpus.py -m interactive -p .
```

```cmd
python .\query_large_corpus.py -m automatic -p .
```

**evaluate_large_corpus.py:**

```cmd
python .\evaluate_large_corpus.py -p .
```