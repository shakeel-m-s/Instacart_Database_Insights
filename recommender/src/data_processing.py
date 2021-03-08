from utils import *

# Constants
zip_path = './instacart-market-basket-analysis'
data_path = './data/extracted_dataset'
store_path = './data/train_test_data'

# Data preparation
extract_csv_data(zip_path, data_path)
df_product_frequency, df_user_product_frequency, df_productsPerUser_train = get_training_data(data_path, store_path)
df_productsPerUser_test = get_testing_data(data_path, store_path)
df_aisles, df_departments, df_products = get_category_data(data_path)
