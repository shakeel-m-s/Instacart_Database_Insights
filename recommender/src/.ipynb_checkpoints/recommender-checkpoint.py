from utils import *
from user_product_matrix import *

import sys
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    assert len(sys.argv) == 4
    for i, arg in enumerate(sys.argv):
        if i == 1:
            print(f"User ID: {arg}")
            user_id = int(arg)
        elif i == 2:
            print(f"Similar Users: {arg}")
            similar_user_num = int(arg)
        elif i == 3:
            print(f"Number of recommended products : {arg}")
            recommend_prod_num = int(arg)
            

    # Use tfidf matrix for similarity calculation
    recommended_prods = generate_recommendation(user_id, \
                                                user_product_matrix_train, \
                                                df_productsPerUser_train, \
                                                df_product_frequency, \
                                                similar_user_num, \
                                                recommend_prod_num)
                                            
    df_aisles, df_departments, df_products = get_category_data(data_path)
    df_report_recommend = report_userBased(recommended_prods, df_products, df_departments, df_aisles)
    display(df_report_recommend)

    # Use tfidf matrix for similarity calculation
    recommended_prods_tfidf = generate_recommendation(user_id, \
                                                user_product_tfidf_matrix_train, \
                                                df_productsPerUser_train, \
                                                df_product_frequency, \
                                                similar_user_num, \
                                                recommend_prod_num)
    df_report_recommend_tfidf = report_userBased(recommended_prods_tfidf, df_products, df_departments, df_aisles)
    display(df_report_recommend_tfidf)