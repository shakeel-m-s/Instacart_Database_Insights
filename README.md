# Instacart Data Analyzer and Recommender System
ECE 143 Project Group 17 (Winter 2021)

Authors:

Alan Contreres

Chaitanya Patil

Sanika Patange

Shakeel Mansoor Shaikna

Yijun Yan

## Installation

Requires python 3.7+

Third-party modules:
- numpy 1.19.5
- pandas 1.2.1
- matplotlib 3.3.4
- scikit-learn 0.24.1
... and a few more listed in requirements.txt

Install dependencies
```
pip install requirements.txt
```

## Usage

Need to download the Instacart Market Basket Dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/overview). The data we generated are stored in [data/martixes](https://github.com/chaitanyaspatil/Instacart_Database_Insights/tree/main/recommender/data/matrixes) folder and the results are stored in [results](https://github.com/chaitanyaspatil/Instacart_Database_Insights/tree/main/recommender/results).


## Data Analysis

All plots are generated in 'data_exploration.ipynb'. The module used is analysis, which contains all the plotting functions.
Make sure you have a directory in which you have all the unzipped .csv files found in the dataset. The path to this directory must be given to each of the functions in order to run it.

The directory 'Rough_Notebooks' contains some rough work done for data analysis for documentation purposes.

## recommender/src

In this directory we have all the .py files that are used for cleaning, processing and training our data sets for the recommender system. Each .py file has it's own functionality. To run the `recommender.py`, you will have to add three arguments namely, User ID (1), No. of similar users (20) and No. of recommended products (10).
```
python recommender.py 1 20 10
```
Since, most of the functions are linked it would be easier to run the `recommender_system.ipynb` which illustrates the working of our recommender system and contains all the plots with regards to the recommender system.

### Group17_Assignment5.ipynb contains the homework test cases assigned to this group.
