import pandas as pd
from flask import Blueprint
from flask import request
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD


movies = Blueprint('movies', __name__ , static_folder='static')


# def load_movie_ratings():
#     # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
#     movie_ratings = pd.read_csv("movies/static/ratings.csv")
#     return movie_ratings
#
# def load_movie_names():
#     # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
#     movie_names = pd.read_csv("movies/static/movies.csv")
#     return movie_names
#
# def load_movie_urls():
#     # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
#     movie_urls = pd.read_csv("movies/static/url.csv",names=['movieId', 'url'])
#     return movie_urls

def load_movies_df():
    # dataframe contains all product details above thereshold of 200 (all products with atleast 200 ratings)
    movies_df = pd.read_csv("movies/static/df_movies.csv")
    return movies_df


# movie_ratings_df = load_movie_ratings()
# movie_names_df = load_movie_names()
# movie_urls_df = load_movie_urls()

df_movies = load_movies_df()
# df_movies = movie_ratings_df.merge(movie_names_df,how='inner',on='movieId').drop(['timestamp','genres'],axis=1)
# df_movies = df_movies.merge(movie_urls_df,how='inner', on='movieId')

# df_movies = pd.merge(movie_ratings_df,movie_names_df).drop(['timestamp','genres'],axis=1)
# df_movies = pd.merge(df_movies,movie_urls_df)
# df_movies.head()

product_ratings_movies = df_movies

def movie_name(prodid):
    return df_movies.loc[df_movies['movieId'] == prodid].iloc[0]['title']


def movie_img(prodid):
    return df_movies.loc[df_movies['movieId'] == prodid].iloc[0]['poster_url']


#API endpoint to return the data needed for autocomplete / It is all the product data above threshold
@movies.route('/getdata', methods=['GET'])
def get_autocomplete_data_sports():
    dups_removed = product_ratings_movies.sort_values("title")
    dups_removed.drop('url',inplace=True,axis=1)
    dups_removed.rename(columns={"poster_url": "url"},inplace=True)
    # dropping ALL duplicte values
    dups_removed.drop_duplicates(subset="title",
                         keep=False, inplace=True)

    return dups_removed.to_json(orient='records')


@movies.route('/getpopular', methods=['GET'])
def getpopular():
    ratings_sum = pd.DataFrame(product_ratings_movies.groupby(['movieId'])['rating'].sum()).rename(
        columns={'rating': 'ratings_sum'})
    top10 = ratings_sum.sort_values('ratings_sum', ascending=False).head(10)

    top10_popular = top10.merge(product_ratings_movies, left_index=True, right_on='movieId').drop_duplicates(
        ['movieId', 'title'])[['movieId', 'title', 'ratings_sum']]
    top10_popular['url'] = top10_popular['movieId'].apply(movie_img)
    return top10_popular.to_json(orient='records')



def get_all_predictions(predictions):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    return top_n

#Uses SVD algorithm
@movies.route('/getrecommendations', methods=['POST'])
def get_predictions():
    #items = [(3, 3), (6, 5), (7, 5)]
    #(item_id,rating)
    content = request.get_json()
    items = content["all_products"]
    user_id = 2323

    df = df_movies
    df = df.dropna()
    newitems = []
    for product in items:
        newitems.append((user_id, product['prodId'], product['rating'], "", "",""))

    df1 = pd.DataFrame(newitems, columns=['userId', 'movieId', 'rating', 'title', 'url','poster_url'])

    df = df.append(df1, ignore_index=True, sort=False)

    counts1 = df['userId'].value_counts()
    counts = df['movieId'].value_counts()

    df1 = df[df['userId'].isin(counts1[counts1 >= 0].index)]
    df1 = df1[df1['movieId'].isin(counts[counts >= 0].index)]

    df1 = df1.sort_values(by='rating')
    df1 = df1.reset_index(drop=True)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df1[['userId', 'movieId', 'rating']], reader)
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
    # To get top 5 reommendation
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

    recommendations_df = pd.DataFrame(recommended_list, columns=['movieId', 'rating'])
    recommendations_df['title'] = recommendations_df['movieId'].apply(movie_name)
    recommendations_df['url'] = recommendations_df['movieId'].apply(movie_img)
    return recommendations_df.to_json(orient='records')


