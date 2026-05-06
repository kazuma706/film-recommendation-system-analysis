Popularity Bias in Collaborative Filtering — MovieLens 25M
==========================================================

This project looks at how recommender systems trained on MovieLens data end up
systematically over-recommending popular films at the expense of niche ones. We
build several models (SVD, content-based TF-IDF, a hybrid, and EASE), measure
the bias, and test whether popularity penalties or sparsity-aware model
selection can fix it.


DATASET
-------
Uses the MovieLens 25M dataset (25 million ratings, ~62,000 films).
Download from: https://grouplens.org/datasets/movielens/25m/
Unzip and place the CSV files (ratings.csv, movies.csv, tags.csv,
genome_scores.csv, genome_tags.csv, links.csv) in a folder called ml-25m/
at the same level as this project folder:

    film data analysis/
        ml-25m/
            ratings.csv
            movies.csv
            ...
        popularity-bias-project/
            notebooks/
            outputs/


NOTEBOOKS (run in this order)
------------------------------
01_eda.ipynb
    Exploratory analysis of the rating distribution. Shows the long-tail
    power law, the 80/20 head/tail split, and which genres are most suppressed.

02_models.ipynb
    Trains an SVD model (scikit-surprise) and a popularity-penalized variant.
    Saves the trained SVD to outputs/svd_model.pkl.

03_fairness_analysis.ipynb
    Evaluates catalog coverage, long-tail coverage, and mean popularity rank
    for SVD vs. penalized SVD. Includes a lambda sweep and Mann-Whitney test.
    NOTE: must be run in the same kernel session as 02_models.ipynb.

04_content_based.ipynb
    Builds a TF-IDF content-based recommender from genres and user tags.
    Saves the model to outputs/content_based_model.pkl.

05_sparsity_experiment.ipynb
    Runs both models across 6 sparsity levels to find where content-based
    beats SVD as data gets sparse. Includes crossover analysis.

04_interactive_viz.ipynb
    Interactive Plotly dashboards: NDCG vs. sparsity curves, accuracy-fairness
    tradeoff, genre suppression chart, and a full model comparison dashboard.
    Loads precomputed CSVs from outputs/ — does not retrain anything.

05_ease.ipynb
    Implements EASE (Embarrassingly Shallow Autoencoder) from scratch and
    runs it through the same two sparsity experiments for comparison.


HOW TO RUN
----------
1. Install requirements (see below).
2. Download the dataset and place CSVs in ml-25m/ as described above.
3. Run notebooks 01 through 05 in order. Note that 03 must share a kernel
   session with 02. The two "04_" and "05_" notebooks can be run after the
   main sequence — they load from saved outputs.


REQUIREMENTS
------------
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
scikit-surprise
scipy
