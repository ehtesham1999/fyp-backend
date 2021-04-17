import numpy as np
import pandas as pd
from flask import Blueprint
from flask import request
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD
nb_closest_images = 5

electronics = Blueprint('electronics', __name__ , static_folder='static')

def load_recommendations():
    # correlation matrix of item similarirt by ratings
    item_similarity_df = pd.read_csv("electronics/static/prod_similarity.csv", index_col=0)
    print("item_similarity_df cached in memory")
    return item_similarity_df


def load_products():
    # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
    product_ratings = pd.read_csv("electronics/static/prod_ratings.csv")
    return product_ratings


def load_images():
    # correlation matrix of image similarity
    images = pd.read_csv('electronics/static/img_similarity.csv', index_col=0)
    return images


item_similarity_df = load_recommendations()
product_ratings = load_products()
cos_similarities_df = load_images()


def prod_name(prodid):
    return product_ratings.loc[product_ratings['prod_ID'] == prodid].iloc[0]['prod_name']


def prod_img(prodid):
    return product_ratings.loc[product_ratings['prod_ID'] == prodid].iloc[0]['imgurl']


@electronics.route('/getsearchimgdata', methods=['GET'])
def get_img_search_data():
    data = {'prod_ID': cos_similarities_df.index}
    search_img_data = pd.DataFrame(data)
    search_img_data['imgurl'] = search_img_data['prod_ID'].apply(prod_img)
    search_img_data['prodname'] = search_img_data['prod_ID'].apply(prod_name)

    return search_img_data.to_json(orient='records')


# function to retrieve the most similar products for a given one
@electronics.route('/getsimilarimage', methods=['POST'])
def retrieve_most_similar_products():
    content = request.get_json()
    given_img = content['prod_ID']
    print("-----------------------------------------------------------------------")
    print("original product:")
    print(given_img)

    print("-----------------------------------------------------------------------")
    print("most similar products:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1]

    print(closest_imgs)
    print(closest_imgs_scores)
    data = {'prod_ID': closest_imgs,
            'scores': closest_imgs_scores}
    img_df = pd.DataFrame(data)

    img_df['imgurl'] = img_df['prod_ID'].apply(prod_img)
    img_df['prodname'] = img_df['prod_ID'].apply(prod_name)

    return img_df.to_json(orient='records')


def get_similar_products(prod_name, user_rating):
    try:
        similar_score = item_similarity_df[prod_name] * (user_rating - 2.5)
        similar_prods = similar_score.sort_values(ascending=False)
    except:
        print("don't have product in model")
        similar_prods = pd.Series([])

    return similar_prods


# API endpoint to return the data needed for autocomplete / It is all the product data above threshold
@electronics.route('/getdata', methods=['GET'])
def get_autocomplete_data():
    dups_removed = product_ratings.sort_values("prod_name")

    # dropping ALL duplicte values
    dups_removed.drop_duplicates(subset="prod_name",
                                 keep=False, inplace=True)

    return dups_removed.to_json(orient='records')


# API endpoint to get top 10 popular products
@electronics.route('/getpopular', methods=['GET'])
def getpopular():
    ratings_sum = pd.DataFrame(product_ratings.groupby(['prod_ID'])['rating'].sum()).rename(
        columns={'rating': 'ratings_sum'})
    top10 = ratings_sum.sort_values('ratings_sum', ascending=False).head(10)

    top10_popular = top10.merge(product_ratings, left_index=True, right_on='prod_ID').drop_duplicates(
        ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name', 'ratings_sum']]
    top10_popular['imgurl'] = top10_popular['prod_ID'].apply(prod_img)
    return top10_popular.to_json(orient='records')

#API endpoint gets recommendations using KNN algorithm
# @electronics.route('/getrecommendations', methods=['POST'])
# def get_recommendations():
#     content = request.get_json()
#     all_products = content["all_products"]
#     similar_products = pd.DataFrame()
#
#     for product in all_products:
#         print(product["prodId"], product["rating"])
#         similar_items = similar_products.append(get_similar_products(product["prodId"], product["rating"]),
#                                                 ignore_index=True)
#
#     # all_recommend = similar_items.sum().sort_values(ascending=False)
#
#     final_results = pd.DataFrame(similar_items.sum().sort_values(ascending=False).head(20))
#     final_results.reset_index(inplace=True)
#     final_results.columns = ['prodid', 'rating']
#     final_results['prodname'] = final_results['prodid'].apply(prod_name)
#     final_results['imgurl'] = final_results['prodid'].apply(prod_img)
#
#     return final_results.to_json(orient='records')

#function to get recommendations using SVD algorithm
def get_all_predictions(predictions):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    return top_n


#API end point gets recommendations using SVD algorithm
#i have set a default userID for getting recommendations
@electronics.route('/getrecommendations', methods=['POST'])
def get_predictions():
    content = request.get_json()
    items = content["all_products"]
    #user_id = content["user_id"]
    user_id = "A5JLAU2ARJ0XX"
    # product_ratings=pd.read_csv('./prod_ratings.csv')
    df = product_ratings
    df = df.dropna()
    newitems = []

    for product in items:
        newitems.append((user_id, product['prodId'], product['rating'], "", ""))

    df1 = pd.DataFrame(newitems, columns=['userID', 'prod_ID', 'rating', 'imgurl', 'prod_name'])

    df = df.append(df1, ignore_index=True, sort=False)

    counts1 = df['userID'].value_counts()
    counts = df['prod_ID'].value_counts()

    df1 = df[df['userID'].isin(counts1[counts1 >= 0].index)]
    df1 = df1[df1['prod_ID'].isin(counts[counts >= 0].index)]

    df1 = df1.sort_values(by='rating')
    df1 = df1.reset_index(drop=True)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df1[['userID', 'prod_ID', 'rating']], reader)
    params = {'n_factors': 35, 'n_epochs': 25, 'lr_all': 0.008, 'reg_all': 0.08}

    # data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()  # Build on entire data set
    algo = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'],
               reg_all=params['reg_all'])
    algo.fit(trainset)

    # Predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()

    # Predicting the ratings for testset
    predictions = algo.test(testset)

    all_pred = get_all_predictions(predictions)
    # To get top 4 reommendation
    n = 5

    for uid, user_ratings in all_pred.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        all_pred[uid] = user_ratings[:n]

    tmp = pd.DataFrame.from_dict(all_pred)
    tmp_transpose = tmp.transpose()
    results = tmp_transpose.loc[user_id]

    recommended_list = []
    for prod_id, rating in results:
        recommended_list.append((prod_id, rating))

    recommendations_df = pd.DataFrame(recommended_list, columns=['prodid', 'rating'])
    recommendations_df['prodname'] = recommendations_df['prodid'].apply(prod_name)
    recommendations_df['imgurl'] = recommendations_df['prodid'].apply(prod_img)
    return recommendations_df.to_json(orient='records')

