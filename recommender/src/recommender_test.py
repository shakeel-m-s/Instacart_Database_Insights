from utils import *
from user_product_matrix import *
from recommender import *

def get_recall(rec, tru):
    """Recommendation recall: |{R & P}|/|P| (R - recommended products, P - relevant products)"""
    return len(rec & tru)/len(tru) if len(tru) != 0 else 0
    
def get_precision(rec, tru):
    """Recommendation precision: |{R & P}|/|R| (R - recommended products, P - relevant products)"""
    return len(rec & tru)/len(rec) if len(rec) != 0 else 0

def test_recommender(feature_matrix, df_productsPerUser_test, df_product_frequency, k, n):
    """Test recommender function. (recall and precision)"""
    right_cases, total_cases = 0, 0
    users = df_productsPerUser_test['user_id'].to_list()[:10000] 
    
    # Variables used for recording
    right_cases, total_cases = 0, 0
    recall_sum, precision_sum = 0, 0
    
    for user in tqdm(users):
        # the user-based recommendation list
        recommended_prods = generate_recommendation(user, feature_matrix, df_productsPerUser_test, df_product_frequency, k, n)
        recommended_prods = set([x for (x, _) in recommended_prods])
        # actual product list
        actual_prods = df_productsPerUser_test[df_productsPerUser_test['user_id'] == user].product_id.tolist()[0]
        if type(actual_prods) == str:
            actual_prods = literal_eval(actual_prods)
        # Check how many right products we recommend
        recall_sum += get_recall(recommended_prods, actual_prods)
        precision_sum += get_precision(recommended_prods, actual_prods)
        right_cases += len(recommended_prods & actual_prods)
        total_cases += len(actual_prods)
    
    # Get average and total value
    print_success('average: (recall, precision) = (%f, %f)' % (recall_sum/len(users), precision_sum/len(users)))
    print_success('total: (recall, precision) = (%f, %f)' % (right_cases/total_cases, right_cases/(n*len(users))))

df_productsPerUser_test = get_testing_data(data_path, store_path)

# Constants
similar_user_num = 20 
recommend_prod_num = 10 

# Testing of the users with id from 1 to 100 (directly using frequeny matrix as feature matrix)
test_recommender(user_product_matrix_train, 
                 df_productsPerUser_test, 
                 df_product_frequency, 
                 similar_user_num, 
                 recommend_prod_num)

# Testing of the users with id from 1 to 100 (using tf-idf matrix as feature matrix)
test_recommender(user_product_tfidf_matrix_train, 
                 df_productsPerUser_test, 
                 df_product_frequency, 
                 similar_user_num, 
                 recommend_prod_num)