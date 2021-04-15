from flask import Flask
import pandas as pd
from flask import request
from scipy.sparse.linalg import svds
import numpy as np

app = Flask(__name__)
nb_closest_images = 5


@app.route('/', methods=['GET'])
def get_img_search_data():
    return "<h1>Welcome</h1>"


def load_recommendations():
    # correlation matrix of item similarirt by ratings
    item_similarity_df = pd.read_csv("static/prod_similarity.csv", index_col=0)
    print("item_similarity_df cached in memory")
    return item_similarity_df


def load_products():
    # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
    product_ratings = pd.read_csv("static/prod_ratings.csv")
    return product_ratings


def load_images():
    # correlation matrix of image similarity
    images = pd.read_csv('static/img_similarity.csv', index_col=0)
    return images


item_similarity_df = load_recommendations()
product_ratings = load_products()
cos_similarities_df = load_images()


def prod_name(prodid):
    return product_ratings.loc[product_ratings['prod_ID'] == prodid].iloc[0]['prod_name']


def prod_img(prodid):
    return product_ratings.loc[product_ratings['prod_ID'] == prodid].iloc[0]['imgurl']


# This is an API endpoint to get image dataframe for all images to recommend products by image similarity
@app.route('/electronics/getsearchimgdata', methods=['GET'])
def get_img_search_data():
    data = {'prod_ID': cos_similarities_df.index}
    search_img_data = pd.DataFrame(data)
    search_img_data['imgurl'] = search_img_data['prod_ID'].apply(prod_img)
    search_img_data['prodname'] = search_img_data['prod_ID'].apply(prod_name)

    return search_img_data.to_json(orient='records')


# function to retrieve the most similar products for a given one
@app.route('/electronics/getsimilarimage', methods=['POST'])
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
@app.route('/electronics/getdata', methods=['GET'])
def get_autocomplete_data():
    dups_removed = product_ratings.sort_values("prod_name")

    # dropping ALL duplicte values
    dups_removed.drop_duplicates(subset="prod_name",
                                 keep=False, inplace=True)

    return dups_removed.to_json(orient='records')


# API endpoint to get top 10 popular products
@app.route('/electronics/getpopular', methods=['GET'])
def getpopular():
    ratings_sum = pd.DataFrame(product_ratings.groupby(['prod_ID'])['rating'].sum()).rename(
        columns={'rating': 'ratings_sum'})
    top10 = ratings_sum.sort_values('ratings_sum', ascending=False).head(10)

    top10_popular = top10.merge(product_ratings, left_index=True, right_on='prod_ID').drop_duplicates(
        ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name', 'ratings_sum']]
    top10_popular['imgurl'] = top10_popular['prod_ID'].apply(prod_img)
    return top10_popular.to_json(orient='records')


@app.route('/electronics/getrecommendations', methods=['POST'])
def get_recommendations():
    content = request.get_json()
    all_products = content["all_products"]
    similar_products = pd.DataFrame()

    for product in all_products:
        print(product["prodId"], product["rating"])
        similar_items = similar_products.append(get_similar_products(product["prodId"], product["rating"]),
                                                ignore_index=True)

    # all_recommend = similar_items.sum().sort_values(ascending=False)

    final_results = pd.DataFrame(similar_items.sum().sort_values(ascending=False).head(20))
    final_results.reset_index(inplace=True)
    final_results.columns = ['prodid', 'rating']
    final_results['prodname'] = final_results['prodid'].apply(prod_name)
    final_results['imgurl'] = final_results['prodid'].apply(prod_img)

    return final_results.to_json(orient='records')


def recommend_it(predictions_df, itm_df, original_ratings_df, num_recommendations=10, ruserId='A108EEYSHGDL6O'):
    # Get and sort the user's predictions

    sorted_user_predictions = predictions_df.loc[ruserId].sort_values(ascending=False)

    # Get the user's data and merge in the item information.
    user_data = original_ratings_df[original_ratings_df.userID == ruserId]
    prev_ratings = pd.DataFrame(user_data).reset_index()
    # print(prev_ratings.head())
    user_full = (user_data.merge(itm_df, how='left', left_on='prod_ID', right_on='prod_ID').
                 sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already purchased {1} items.'.format(ruserId, user_full.shape[0]))
    print('Recommending the highest {0} predicted  items not already purchased.'.format(num_recommendations))

    # Recommend the highest predicted rating items that the user hasn't bought yet.
    recommendations = (itm_df[~itm_df['prod_ID'].isin(user_full['prod_ID'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='prod_ID',
                                 right_on='prod_ID').
                           rename(columns={ruserId: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )
    topk = \
    recommendations.merge(original_ratings_df, left_index=True, right_on='prod_ID', left_on='prod_ID').drop_duplicates(
        ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name']]

    return [prev_ratings, topk]


@app.route('/electronics/getrecommendations_svd', methods=['POST'])
def get_svd_recommendations():
    content = request.get_json()
    items = content["all_products"]
    user_id = content["user_id"]

    df = product_ratings
    df = df.dropna()

    counts1 = df['userID'].value_counts()
    counts = df['prod_ID'].value_counts()

    df1 = df[df['userID'].isin(counts1[counts1 >= 10].index)]
    df1 = df1[df1['prod_ID'].isin(counts[counts >= 10].index)]

    df1 = df1.sort_values(by='rating')
    df1 = df1.reset_index(drop=True)
    count_users = df1.groupby("userID", as_index=False).count()

    count = df1.groupby("prod_ID", as_index=False).mean()

    items_df = count[['prod_ID']]
    users_df = count_users[['userID']]

    df_clean_matrix = df1.pivot(index='prod_ID', columns='userID', values='rating').fillna(0)
    df_clean_matrix = df_clean_matrix.T
    df_clean_matrix.loc[user_id][:] = 0

    for product in items:
        df_clean_matrix.loc[user_id][product["prodId"]] = product["rating"]

    R = (df_clean_matrix).to_numpy()

    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=df_clean_matrix.columns)
    preds_df['userID'] = users_df
    preds_df.set_index('userID', inplace=True)
    preds_df.head()

    # 'A11KZ906QD08C5'
    final_results = recommend_it(preds_df, items_df, df1, 5, user_id)
    final_results = final_results[1]
    final_results.reset_index(inplace=True)
    final_results['imgurl'] = final_results['prod_ID'].apply(prod_img)
    final_results['rating'] = final_results['prod_ID'].apply(lambda prodid: preds_df.loc[user_id][prodid])
    return final_results.to_json(orient='records')


if __name__ == '__main__':
    app.run()
