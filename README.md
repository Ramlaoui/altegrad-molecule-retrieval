# Altegrad Challenge 2023 - Molecule Retrieval with NLP Queries

## Introduction

The goal of this challenge is to retrieve molecules from a database given a natural language query. The database is composed of graphs representing molecules, and the queries are sentences describing the desired molecules. The task is to retrieve the molecules that are the most relevant to the query, using the graph structure available.

## Data

The dataset is composed of two parts: the molecules and the queries. The molecules are represented as graphs, and the queries are sentences. The molecules are provided in the form of a graph, where each node represents an atom and each edge represents a bond. The queries are sentences describing the desired molecules. 

The dataset is split into a training, validation and testing set. The training set contains around 25k molecules and queries, the validation set contains around 3k molecules and queries, as well as the testing set.

## Evaluation

The evaluation metric is the Mean Reciprocal Rank (MRR). The MRR is a metric used to evaluate systems that return a list of possible responses to a query, ordered by probability. The MRR is the average of the reciprocal ranks of results for a sample of queries Q:

\[
MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
\]

where \(rank_i\) is the rank of the first relevant result for query \(i\).

## Submission

The submission file can be generated using the method `submit_run` from the trainer class. This method used a trained model that was run using the same trainer class and runs it on the test set in order to generate a pandas DataFrame following the structure of the challenge with the cosine similarity between the queries and the molecules. The DataFrame is then saved as a csv file.

## Code

The `src` directory contains the main files to implement the core of the pipeline and the models. Most of the important functions are contained in the`trainer.py` file. The `models` directory contains the implementation of the models used in the pipeline. The `data` directory contains the implementation of the dataset and the dataloaders. It is possible to use wandb to log the training process and the results of the models.

## Configs

The `configs` directory contains the configuration files used to train the models. For every model a specific config file is used to define the hyperparameters and the data used in the training process.

