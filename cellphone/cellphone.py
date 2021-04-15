from flask import Blueprint
import pandas as pd
from flask import request
from scipy.sparse.linalg import svds
import numpy as np

nb_closest_images = 5

cellphone = Blueprint('cellphone', __name__ , static_folder='static')

def load_products_cellphone():
    # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
    product_ratings = pd.read_csv("cellphone/static/prod_ratings_cellphone.csv")
    return product_ratings

def load_recommendations_cellphone():
    item_similarity_df = pd.read_csv("cellphone/static/prod_similarity_cellphone.csv", index_col=0)
    print("item_similarity_df cached in memory")
    return item_similarity_df

def load_images_cellphone():
    # correlation matrix of image similarity
    images = pd.read_csv('cellphone/static/img_similarity_cellphone.csv',index_col=0)
    return images

item_similarity_df_cellphone = load_recommendations_cellphone()
product_ratings_cellphone = load_products_cellphone()
cos_similarities_df_cellphone = load_images_cellphone()


def prod_name_cellphone(prodid):
    return product_ratings_cellphone.loc[product_ratings_cellphone['prod_ID'] == prodid ].iloc[0]['prod_name']

def prod_img_cellphone(prodid):
    return product_ratings_cellphone.loc[product_ratings_cellphone['prod_ID'] == prodid ].iloc[0]['imgurl']



#This is an API endpoint to get image dataframe for all images to recommend products by image similarity
@cellphone.route('/getsearchimgdata', methods=['GET'])
def get_img_search_data_cellphone():
    data = {'prod_ID': cos_similarities_df_cellphone.index}
    search_img_data = pd.DataFrame(data)
    search_img_data['imgurl'] = search_img_data['prod_ID'].apply(prod_img_cellphone)
    search_img_data['imgurl'] = search_img_data['imgurl'].apply(lambda x: x.strip('][').split(',')[0])
    search_img_data['imgurl'] = search_img_data['imgurl'].apply(lambda x: x[1:-1])
    search_img_data['prodname'] = search_img_data['prod_ID'].apply(prod_name_cellphone)

    return search_img_data.to_json(orient='records')




# function to retrieve the most similar products for a given one
@cellphone.route('/getsimilarimage', methods=['POST'])
def retrieve_most_similar_products_cellphone():
    content = request.get_json()
    given_img = content['prod_ID']
    print("-----------------------------------------------------------------------")
    print("original product:")
    print(given_img)



    print("-----------------------------------------------------------------------")
    print("most similar products:")

    closest_imgs = cos_similarities_df_cellphone[given_img].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cos_similarities_df_cellphone[given_img].sort_values(ascending=False)[1:nb_closest_images + 1]

    print(closest_imgs)
    print(closest_imgs_scores)
    data = {'prod_ID': closest_imgs,
            'scores': closest_imgs_scores}
    img_df = pd.DataFrame(data)

    img_df['imgurl'] = img_df['prod_ID'].apply(prod_img_cellphone)
    img_df['imgurl'] = img_df['imgurl'].apply(lambda x: x.strip('][').split(',')[0])
    img_df['imgurl'] = img_df['imgurl'].apply(lambda x: x[1:-1])

    img_df['prodname'] = img_df['prod_ID'].apply(prod_name_cellphone)

    return img_df.to_json(orient='records')



def get_similar_products_cellphone(prod_name, user_rating):
    try:
        similar_score = item_similarity_df_cellphone[prod_name] * (user_rating - 2.5)
        similar_prods = similar_score.sort_values(ascending=False)
    except:
        print("don't have product in model")
        similar_prods = pd.Series([])

    return similar_prods


#API endpoint to return the data needed for autocomplete / It is all the product data above threshold
@cellphone.route('/getdata', methods=['GET'])
def get_autocomplete_data_cellphone():
    dups_removed = product_ratings_cellphone.sort_values("prod_name")

    # dropping ALL duplicte values
    dups_removed.drop_duplicates(subset="prod_name",
                         keep=False, inplace=True)
    dups_removed['imgurl'] = dups_removed['imgurl'].apply(lambda x: x.strip('][').split(',')[0])
    dups_removed['imgurl'] = dups_removed['imgurl'].apply(lambda x: x[1:-1])
    return dups_removed.to_json(orient='records')


@cellphone.route('/getpopular', methods=['GET'])
def getpopular_cellphone():
    ratings_sum = pd.DataFrame(product_ratings_cellphone.groupby(['prod_ID'])['rating'].sum()).rename(columns={'rating': 'ratings_sum'})
    top10 = ratings_sum.sort_values('ratings_sum', ascending=False).head(10)

    top10_popular = top10.merge(product_ratings_cellphone, left_index=True, right_on='prod_ID').drop_duplicates(
        ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name', 'ratings_sum']]
    top10_popular['imgurl'] = top10_popular['prod_ID'].apply(prod_img_sports)
    top10_popular['imgurl'] = top10_popular['imgurl'].apply(lambda x: x.strip('][').split(',')[0])
    top10_popular['imgurl'] = top10_popular['imgurl'].apply(lambda x: x[1:-1])

    return top10_popular.to_json(orient='records')


@cellphone.route('/getrecommendations', methods=['POST'])
def get_recommendations_cellphone():
    content = request.get_json()
    all_products = content["all_products"]
    similar_products = pd.DataFrame()

    for product in all_products:
        print(product["prodId"], product["rating"])
        similar_items = similar_products.append(get_similar_products_cellphone(product["prodId"], product["rating"]), ignore_index=True)


    #all_recommend = similar_items.sum().sort_values(ascending=False)

    final_results = pd.DataFrame(similar_items.sum().sort_values(ascending=False).head(20))
    final_results.reset_index(inplace=True)
    final_results.columns = ['prodid', 'rating']
    final_results['prodname'] = final_results['prodid'].apply(prod_name_cellphone)
    final_results['imgurl'] = final_results['prodid'].apply(prod_img_cellphone)
    final_results['imgurl'] = final_results['imgurl'].apply(lambda x: x.strip('][').split(', ')[0])
    final_results['imgurl'] = final_results['imgurl'].apply(lambda x: x[1:-1])



    return final_results.to_json(orient='records')

