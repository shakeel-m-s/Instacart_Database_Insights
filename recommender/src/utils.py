from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import os
import heapq
import zipfile
import pandas as pd
import numpy as np
import scipy.sparse as sparse

# Some logistics helping functions
class colors:
    """Color used for printing."""
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_success(message):
    """Specific print func for this notebook."""
    assert isinstance(message, str)
    print(f"{colors.HEADER}[Recommender Message]{colors.ENDC} - {colors.OKGREEN}{message}{colors.ENDC}")


def print_failure(message):
    """Specific error func for this notebook."""
    assert isinstance(message, str)
    print(f"{colors.HEADER}[Recommender Failure]{colors.ENDC} - {colors.FAIL}{message}{colors.ENDC}")


def print_warning(message):
    """Specific warning func for this notebook."""
    assert isinstance(message, str)
    print(f"{colors.HEADER}[Recommender Warning]{colors.ENDC} - {colors.WARNING}{message}{colors.ENDC}")


def extract_csv_data(zip_path, data_path):
    """Extract and retrieve csv data."""
    assert isinstance(zip_path, str) and isinstance(data_path, str)
    # Get files in the zip_path
    zip_files = [os.path.join(zip_path, f) for f in os.listdir(zip_path)]
    print_success('Files in ' + zip_path + ':\n' + str(zip_files))
    
    # Pass if data_path already exists
    if os.path.exists(data_path):  ### Path may exist but not the data... Its better to rewrite the data
        print_warning('Extracted data (%s) are already existed.' % data_path)
        return
    
    # Store the extracted csv files in data_path
    for zip_file in zip_files:
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_path)
    print_success('All zip files are extracted.')


def get_training_data(data_path, store_path):
    """Define the prior(eval_set="prior") orders as the training dataset."""
    assert isinstance(data_path, str) and isinstance(store_path, str)
    # Filenames for storing the processed data
    prodsPerUser_filename = os.path.join(store_path, 'productsPerUser_train.csv')
    userProdFreq_filename = os.path.join(store_path, 'user_product_frequency_train.csv')
    prodFreq_filename = os.path.join(store_path, 'product_frequency_train.csv')
    
    # Pass if files are already existed
    if os.path.exists(prodsPerUser_filename) and os.path.exists(userProdFreq_filename) and os.path.exists(prodFreq_filename):
        print_warning('Training data are already existed.')
        return pd.read_csv(prodFreq_filename), \
            pd.read_csv(userProdFreq_filename), \
            pd.read_csv(prodsPerUser_filename)
    
    try:
        # Load the csv files as dataframes
        df_orders = pd.read_csv(os.path.join(data_path, 'orders.csv'))
        df_order_products_train = pd.read_csv(os.path.join(data_path, 'order_products__prior.csv'))
        
        # Trim the unnecessary columns
        df_order_products_train = df_order_products_train[["order_id", "product_id"]]
        
        # Get the frequency of occurrence for each product (ready for tf-idf)
        df_product_frequency = df_order_products_train['product_id'].value_counts()
        df_product_frequency = df_product_frequency.rename_axis('product_id').reset_index(name='frequency')
        print_success('Calculation of product frequency is completed.')
        
        # Get the direct relation between products and users
        df_usersAndProducts_train = pd.merge(df_orders, df_order_products_train, on='order_id', how='inner')
        df_usersAndProducts_train = df_usersAndProducts_train[['user_id', 'product_id']]
        df_productsPerUser_train = df_usersAndProducts_train.groupby('user_id').agg(set).reset_index()
        print_success('Calculation of productsPerUser is completed.')
        
        # Get the frequency of occurence for each user-product pair
        df_user_product_frequency = df_usersAndProducts_train.groupby(['user_id', 'product_id'])\
            .size().reset_index().rename(columns={0: 'frequency'})
        print_success('Calculation of user-product-pair frequency is completed.')
        
        # Store the processed data to enhance efficiency
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        df_productsPerUser_train.to_csv(prodsPerUser_filename, index_label=False)
        df_user_product_frequency.to_csv(userProdFreq_filename, index_label=False)
        df_product_frequency.to_csv(prodFreq_filename, index_label=False)
        
        print_success('Training data are retrieved and saved.')
        return df_product_frequency, df_user_product_frequency, df_productsPerUser_train
    except Exception as e: 
        print_failure(str(e))

        
def get_testing_data(data_path, store_path):
    """Define the current(eval_set="train") orders as the testing dataset."""
    assert isinstance(data_path, str) and isinstance(store_path, str)
    # Filename for testing the recommender system
    test_filename = os.path.join(store_path, 'productsPerUser_test.csv')
    
    # Pass if file is already existed
    if os.path.exists(test_filename):
        print_warning('Testing data are already existed.')
        return pd.read_csv(test_filename)
    
    try:
        # Load the csv files as dataframes
        df_orders = pd.read_csv(os.path.join(data_path, 'orders.csv'))
        df_order_products_test = pd.read_csv(os.path.join(data_path, 'order_products__train.csv'))
        
        # Trim the unnecessary columns
        df_order_products_test = df_order_products_test[["order_id", "product_id"]]
        
        # Get the direct relation between products and users
        df_usersAndProducts_test = pd.merge(df_orders, df_order_products_test, on='order_id', how='inner')
        df_usersAndProducts_test = df_usersAndProducts_test[['user_id', 'product_id']]
        df_productsPerUser_test = df_usersAndProducts_test.groupby('user_id').agg(set).reset_index()
        
        # Store the processed data to enhance efficiency
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        df_productsPerUser_test.to_csv(test_filename, index_label=False)
        
        print_success('Testing data are retrieved and saved.')
        return df_productsPerUser_test
    except Exception as e: 
        print_failure(str(e))

        
def get_category_data(data_path):
    """Get the other category csv datasets."""
    assert isinstance(data_path, str)
    try:
        df_aisles = pd.read_csv(os.path.join(data_path, 'aisles.csv'))
        df_departments = pd.read_csv(os.path.join(data_path, 'departments.csv'))
        df_products = pd.read_csv(os.path.join(data_path, 'products.csv'))
        print_success('Category data are retrieved.')
        return df_aisles, df_departments, df_products
    except Exception as e:
        print_failure(str(e))

