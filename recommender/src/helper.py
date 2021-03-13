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
import sys
import matplotlib.pyplot as plt

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


def build_user_product_matrix(df_user_product_frequency, matrix_file_path, matrix_name):
    """Build and store coo/csr sparse matrix of user-product matrix."""
    assert isinstance(df_user_product_frequency, pd.DataFrame)
    assert isinstance(matrix_file_path, str) and isinstance(matrix_name, str)
    matrix_path = os.path.join(matrix_file_path, matrix_name)
    if os.path.exists(matrix_path):
        print_warning('User-product matrix is already existed.')
        return sparse.load_npz(matrix_path).tocsr()
    
    df_user_product_frequency['user_id'] = df_user_product_frequency['user_id'].astype('category')
    df_user_product_frequency['product_id'] = df_user_product_frequency['product_id'].astype('category')
    
    # Define sparse user-product matrix in coo format
    data = df_user_product_frequency['frequency']
    row = df_user_product_frequency['user_id'].cat.codes.copy()
    col = df_user_product_frequency['product_id'].cat.codes.copy()
    user_product_matrix = sparse.coo_matrix((data, (row, col)))
    
    # Store and return the sparse matrix
    if not os.path.exists(matrix_file_path):
        os.mkdir(matrix_file_path) 
    sparse.save_npz(matrix_path, user_product_matrix)
    print_success('User-product matrix is stored at %s' % matrix_path)
    return user_product_matrix.tocsr()


def build_tfidf_matrix(tf, matrix_file_path, matrix_name):
    """Build tf-idf sparse matrix for product. 'tf' refers to term frequency."""
    assert isinstance(tf, sparse.csr.csr_matrix)
    assert isinstance(matrix_file_path, str) and isinstance(matrix_name, str)
    matrix_path = os.path.join(matrix_file_path, matrix_name)
    if os.path.exists(matrix_path):
        print_warning('User-product TF-IDF matrix is already existed.')
        return sparse.load_npz(matrix_path).tocsr()

    tf_idf = coo_matrix(tf)
    
    # Get total number of documents (here is user number)
    N = tf.shape[0]
    
    # Calculate IDF (inverse document frequency)
    idf = np.log(N / (1 + np.bincount(tf_idf.col)))
    
    # Since terms don’t show up in many documents, we apply a square root penalty over tf to dampen it.
    tf_idf.data = np.sqrt(tf_idf.data) * idf[tf_idf.col] 

    # Store and return the sparse matrix
    if not os.path.exists(matrix_file_path):
        os.mkdir(matrix_file_path) 
    sparse.save_npz(matrix_path, tf_idf)
    print_success('User-product TF-IDF matrix is stored at %s' % matrix_path)
    return tf_idf.tocsr()


# User-based recommendation
def get_topK_similar_users(user_id, feature_matrix, k):
    """Find the most k similar users based on similarity."""
    assert isinstance(user_id, int) and isinstance(k, int) 
    assert isinstance(feature_matrix, sparse.csr.csr_matrix)
    # Get list of cosine similarities
    similarities = cosine_similarity(feature_matrix, feature_matrix[user_id - 1], False)
    
    # Select top K similar users
    top_K_similar_users = heapq.nlargest(k + 1, range(similarities.shape[0]), similarities.toarray().take)[1:]
    top_K_similar_users = [x + 1 for x in top_K_similar_users]
    
    # Return the list excluding the target user
    return top_K_similar_users


def generate_recommendation(user_id, feature_matrix, df_productsPerUser, df_product_frequency, k, n):
    """Find the most n recommended products based on the shopping history of the similar users."""
    assert isinstance(user_id, int) and isinstance(k, int) and isinstance(n, int) 
    assert isinstance(feature_matrix, sparse.csr.csr_matrix)
    assert isinstance(df_product_frequency, pd.DataFrame) and isinstance(df_productsPerUser, pd.DataFrame)
    # Get top k similar users
    topK_similar_users = get_topK_similar_users(user_id, feature_matrix, k)
    
    # Product popularity is defined as following 2 parts:
    # 1. the number of similar users who buy this product
    # 2. the buying frequency of this product in all users
    
    recommended_prods = defaultdict(int)
    user_prods = df_productsPerUser['product_id'][df_productsPerUser['user_id'] == user_id].values[0]
    if type(user_prods) == str:
        user_prods = literal_eval(user_prods)
    for user in topK_similar_users:
        prods = df_productsPerUser['product_id'][df_productsPerUser['user_id'] == user].values
        prods = set() if len(prods) == 0 else prods[0]
        if type(prods) == str:
            prods = literal_eval(prods)
        for prod in prods:
            recommended_prods[prod] += 1
    
    # Get popularity for each prod
    recommended_prods = [(p, (x, int(df_product_frequency[df_product_frequency['product_id'] == x].frequency))) \
                         for (p, x) in recommended_prods.items()]
    
    # Sort the products based on the popularity in the set of similar users
    recommended_prods = sorted(recommended_prods, key = lambda kv : (kv[1], kv[0]), reverse=True)
    return recommended_prods[:n]
    

def report_userBased(recommended_prods, df_products, df_departments, df_aisles):
    '''Prints out the details of the recommended products in a dataframe.'''
    assert isinstance(df_products, pd.DataFrame) and isinstance(df_aisles, pd.DataFrame) 
    assert isinstance(df_departments, pd.DataFrame)
    data = {'product_id': [], 'popularity': []}
    for product in recommended_prods:
        data['product_id'].append(product[0])
        data['popularity'].append(product[1])
    df = pd.DataFrame(data, columns=list(data.keys()))
    df = pd.merge(df, df_products, on='product_id', how='inner') # add product details
    df = pd.merge(df, df_departments, on='department_id', how='inner') # add department details
    df = pd.merge(df, df_aisles, on='aisle_id', how='inner') # add aisle details
    return df.sort_values(by='popularity', ascending=False)

