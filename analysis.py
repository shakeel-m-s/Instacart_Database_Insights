def meanOrderFrequency(path_to_dataset):
    """
    Displays the mean order frequency by utilizing the orders table.
    :param path_to_dataset: this path should have all the .csv files for the dataset
    :type path_to_dataset: str
    """
    assert isinstance(path_to_dataset, str)
    import pandas as pd
    order_file_path = path_to_dataset + '/orders.csv'
    orders = pd.read_csv(order_file_path)
    print('On an average, people order once every ', orders['days_since_prior_order'].mean(), 'days')
    
    
def numOrdersVsDays(path_to_dataset):
    """
    Displays the number of orders and how this number varies with change in days since last order.
    :param path_to_dataset: this path should have all the .csv files for the dataset
    :type path_to_dataset: str
    """
    
    assert isinstance(path_to_dataset, str)
    
    import pandas as pd 
    import numpy as np 
    import matplotlib.pyplot as plt 
    import matplotlib
    
    
    
    order_file_path = path_to_dataset + '/orders.csv'
    
    orders = pd.read_csv(order_file_path)
    
    order_by_date = orders.groupby(by='days_since_prior_order').count()
    
    fig = plt.figure(figsize = [15, 7.5])
    ax = fig.add_subplot()
    order_by_date['order_id'].plot.bar(color = '0.75')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize= 15)
    plt.yticks(fontsize=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x))))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x/1000))))
    ax.set_xlabel('Days since previous order', fontsize=16)
    ax.set_ylabel('Number of orders / 1000', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_children()[7].set_color('0.1')
    ax.get_children()[14].set_color('0.1')
    ax.get_children()[21].set_color('0.1')
    ax.get_children()[30].set_color('0.1')
    my_yticks = ax.get_yticks()
    plt.yticks([my_yticks[-2]], visible=True)
    plt.xticks(rotation = 'horizontal');
    
    

def numOrderDaysSizeBubble(path_to_dataset):
    """
    Plots a bubble plot in which:
    x: Days since Previous Order
    y: Number of orders/1000
    size: Average Size of order given it was placed on x
    
    :param path_to_dataset: this path should have all the .csv files for the dataset
    :type path_to_dataset: str
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    
    order_file_path = path_to_dataset + '/orders.csv'
    order_product_prior_file_path = path_to_dataset + '/order_products__prior.csv'
    
    orders = pd.read_csv(order_file_path)
    order_products_prior = pd.read_csv(order_product_prior_file_path)
    
    order_id_count_products = order_products_prior.groupby(by='order_id').count()
    orders_with_count = order_id_count_products.merge(orders, on='order_id')
    order_by_date = orders.groupby(by='days_since_prior_order').count()
    # take above table and group by days_since_prior_order

    df_mean_order_size = orders_with_count.groupby(by='days_since_prior_order').mean()['product_id']
    df_mean_order_renamed = df_mean_order_size.rename('average_order_size')


    bubble_plot_dataframe = pd.concat([order_by_date['order_id'], df_mean_order_renamed], axis=1)

    bubble_plot_dataframe['average_order_size'].index.to_numpy()

    fig = plt.figure(figsize=[15,7.5])
    ax = fig.add_subplot()
    plt.scatter(bubble_plot_dataframe['average_order_size'].index.to_numpy(), bubble_plot_dataframe['order_id'].values, s=((bubble_plot_dataframe['average_order_size'].values/bubble_plot_dataframe['average_order_size'].values.mean())*10)**3.1, alpha=0.5, c = '0.5')

    plt.xticks(np.arange(0, 31, 1.0));
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Days since previous order', fontsize=16)
    ax.set_ylabel('Number of orders / 1000', fontsize=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x))))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x/1000))))
    my_yticks = ax.get_yticks()
    plt.yticks([my_yticks[-2], my_yticks[0]], visible=True);

    fig = plt.figure(figsize=[10,9])
    ax = fig.add_subplot()
    plt.scatter(bubble_plot_dataframe['average_order_size'].index.to_numpy()[:8], bubble_plot_dataframe['order_id'].values[:8], s=((bubble_plot_dataframe['average_order_size'].values[:8]/bubble_plot_dataframe['average_order_size'].values.mean())*10)**3.1, alpha=0.5, c = '0.5')

    plt.xticks(np.arange(0, 8, 1.0));
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Days since previous order', fontsize=16)
    ax.set_ylabel('Number of orders / 1000', fontsize=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x))))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x/1000))))
    my_yticks = ax.get_yticks()
    plt.yticks([my_yticks[-2], my_yticks[0]], visible=True);