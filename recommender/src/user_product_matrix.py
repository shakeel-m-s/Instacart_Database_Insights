from utils import *

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


# Constants
data_path = './data/extracted_dataset'
store_path = './data/train_test_data'
train_matrix_path = './data/matrixes'
train_matrix_name = 'user_product_train.npz'
train_tfidf_matrix_name = 'user_product_tfidf_train.npz'

# From processed training data
df_product_frequency, df_user_product_frequency, df_productsPerUser_train = get_training_data(data_path, store_path)

# Generate sparse matrix for training
user_product_matrix_train = build_user_product_matrix(df_user_product_frequency, train_matrix_path, train_matrix_name)

# Generate tf-idf matrix based on user-product-pair matrix
user_product_tfidf_matrix_train = build_tfidf_matrix(user_product_matrix_train, train_matrix_path, train_tfidf_matrix_name)
