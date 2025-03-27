import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

ratings= pd.read_csv('./data/ratings.csv')
movies = pd.read_csv('./data/movies.csv')
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
movies_with_genres = movies.copy()


def load_process_lens_dataset():

    movies_with_ratings = pd.merge(movies, ratings, on='movieId', how='inner')
    
    average_ratings = movies_with_ratings.groupby('movieId')['rating'].mean()

    popular_movies = average_ratings[average_ratings >= 3.0].index

    filtered_movies_with_ratings = movies_with_ratings[movies_with_ratings['movieId'].isin(popular_movies)]

    # remove non expert users, ie, users with less than 75% of the quantile of ratings, keeping users with more than 89 ratings
    user_rating_counts = filtered_movies_with_ratings.groupby('userId')['rating'].count()
    print(user_rating_counts.describe())
    filtered_users = user_rating_counts[(user_rating_counts >= user_rating_counts.quantile(0.75)) & 
                                        (user_rating_counts <= user_rating_counts.quantile(0.99))].index

    filtered_movies_with_ratings = filtered_movies_with_ratings[filtered_movies_with_ratings['userId'].isin(filtered_users)]
    
    filtered_movies_with_ratings.drop(columns=["timestamp","genres","title"], inplace=True)
    filtered_movies_with_ratings = filtered_movies_with_ratings.drop(columns=["timestamp", "genres", "title"])
    # Removed incomplete and unused line

    # filter out movies with less than x amount of ratings to clean up further
    # after looking up some of the movies around the 70-75th quantile range, these movies are super niche and not very popular, so I will filter out movies with less than 30 ish ratings
    movie_rating_counts = filtered_movies_with_ratings.groupby('movieId').size()
    

    min_ratings_threshold = 30
    popular_movies = movie_rating_counts[movie_rating_counts >= min_ratings_threshold].index


    filtered_movies_with_ratings = filtered_movies_with_ratings[
        filtered_movies_with_ratings['movieId'].isin(popular_movies)
    ]
        
    user_movie_matrix = filtered_movies_with_ratings.pivot(index='userId', columns='movieId', values='rating')
    user_movie_matrix = user_movie_matrix.fillna(0)

    return user_movie_matrix


def scale_sparse_ratings(matrix):

    data = matrix.values.copy()
    rows, cols = data.shape
    
    nonzero_mask = data > 0
    
    # Scale each user's ratings to be between 0 and 1,
    scaler = MinMaxScaler()
    for i in range(rows):
        # Get indices of rated movies for this user
        rated_indices = np.where(nonzero_mask[i])[0]
        if len(rated_indices) > 1:  # Only scale if user has rated multiple movies
            # Extract, reshape for scaler, scale, and put back
            user_ratings = data[i, rated_indices].reshape(-1, 1)
            scaled_ratings = scaler.fit_transform(user_ratings).flatten()
            data[i, rated_indices] = scaled_ratings
    
    # Convert back to DataFrame
    return pd.DataFrame(data, index=matrix.index, columns=matrix.columns)



def prepare_user_ratings(ratings_file_path, user_movie_matrix_columns):
    user_data = pd.read_csv(ratings_file_path)
    user_data = user_data.dropna(subset=['movieId'])

    user_data['scaled_rating'] = (user_data['Rating'] - 1) / 4.0
    
    user_ratings = pd.DataFrame(0, index=[0], columns=user_movie_matrix_columns)
    
    for _, row in user_data.iterrows():
        movie_id = row['movieId']
        if movie_id in user_ratings.columns:
            user_ratings.loc[0, movie_id] = row['scaled_rating']
    
    return user_data, user_ratings


def fit_model(
    user_movie_matrix,
    n_components=50,
    max_iter=200,
    init='nndsvd',
    solver='cd',
    tol=0.0001,
    l1_ratio=0.5,
    sample_size=1.0
):

    print(f"\n--- Fitting model with {sample_size*100:.0f}% of users ---")
    
    # Sample users 
    if sample_size < 1.0:
        n_users = int(user_movie_matrix.shape[0] * sample_size)
        training_matrix = user_movie_matrix.sample(n=n_users, replace=False, random_state=42)
    else:
        training_matrix = user_movie_matrix
    
    print(f"Training matrix shape: {training_matrix.shape}")
    
    # Train the model
    start_time = time.time()
    nmf = NMF(
        n_components=n_components,
        max_iter=max_iter,
        init=init,
        solver=solver,
        tol=tol,
        l1_ratio=l1_ratio
    )
    
    user_factors = nmf.fit_transform(training_matrix)
    item_factors = nmf.components_
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Iterations completed: {nmf.n_iter_}")
    print(f"Final error: {nmf.reconstruction_err_}")
    
    return nmf, training_matrix, training_time


def get_recommendations(
    nmf_model,
    training_matrix,
    my_ratings,
    my_data,
    n_recommendations=20,
    movie_id_to_title_map=None
):

    my_user_factors = nmf_model.transform(my_ratings)
    
    # Generate predictions
    predicted_ratings = np.dot(my_user_factors, nmf_model.components_)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=training_matrix.columns)
    
    # Get movies already rated
    rated_movies = set(int(movie_id) for movie_id in my_data['movieId'].values)
    print(f"You've rated {len(rated_movies)} movies")
    
    # Filter for unrated movies
    unrated_movies = [m for m in training_matrix.columns if int(m) not in rated_movies]
    print(f"Found {len(unrated_movies)} movies you haven't rated")
    
    # Get recommendations
    recommendations = []
    for movie_id in unrated_movies:
        pred_rating = predicted_ratings_df.loc[0, movie_id]
        recommendations.append((movie_id, pred_rating))
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_recommendations = recommendations[:n_recommendations]
    
    # Display recommendations
    if movie_id_to_title_map:
        print("\nTop Recommendations:")
        for movie_id, predicted_rating in top_recommendations:
            # Convert back to original rating scale
            original_scale_rating = predicted_rating * 4 + 1
            
            movie_title = movie_id_to_title_map.get(movie_id, f"Unknown Movie (ID: {movie_id})")
            print(f"Movie: {movie_title}, Predicted Rating: {original_scale_rating:.2f}/5.00")
    
    return top_recommendations


user_movie_matrix = load_process_lens_dataset()
user_movie_matrix_scaled = scale_sparse_ratings(user_movie_matrix)

ludde_data, ludde_ratings = prepare_user_ratings(
    "data/lddec_ratings_with_ids.csv", 
    user_movie_matrix_scaled.columns
)
# charlie_data, charlie_ratings = prepare_user_ratings(
#     "data/chaarll_ratings_with_ids.csv", 
#     user_movie_matrix_scaled.columns
# )
# tilda_data, tilda_ratings = prepare_user_ratings(
#     "data/tilda_h_ratings_with_ids.csv",
#     user_movie_matrix_scaled.columns
# )
# 
# elliott_data, elliott_ratings = prepare_user_ratings(
#     "data/loelliot_ratings_with_ids.csv", 
#     user_movie_matrix_scaled.columns
# )

nmf_model, training_matrix, _ = fit_model(
    user_movie_matrix_scaled,
    n_components=30,
    l1_ratio=0.5,
    sample_size=0.1,
    max_iter=500
)

#%%
recommendations = get_recommendations(
    nmf_model=nmf_model,
    training_matrix=training_matrix,
    my_ratings=ludde_ratings,
    my_data=ludde_data,
    n_recommendations=200,
    movie_id_to_title_map=movie_id_to_title,
)
