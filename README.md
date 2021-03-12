# Instacart Data Analyzer and Recommender System
ECE 143 Project Group 17 (Winter 2021)

## Installation

Requires python 3.7+

Third-party modules:
- numpy 1.19.5
- pandas 1.2.1
- matplotlib 3.3.4
- scikit-learn 0.24.1

Install dependencies
```
pip install requirements.txt
```

## Usage

Need to download the Instacart Market Basket Dataset from Kaggle (https://www.kaggle.com/c/instacart-market-basket-analysis/overview). The data we generated are stored in [data/martixes](https://github.com/chaitanyaspatil/Instacart_Database_Insights/tree/main/recommender/data/matrixes) folder and the results are stored in [results](https://github.com/chaitanyaspatil/Instacart_Database_Insights/tree/main/recommender/results).

## recommender/src

In this directory we have all the .py files that are used for cleaning, processing and training our data sets for the recommender system. Each .py file has it's own functionality. To run the `recommender.py`, you will have to add three arguments namely, User ID (1), No. of similar users(20) and No. of recommended products(10).
```
python recommender.py 1 20 10
```
Since, most of the functions are linked it would be easier to run the `recommender_system.ipynb` which illustrates the working of our recommender system and contains all the plots with regards to the recommender system.

