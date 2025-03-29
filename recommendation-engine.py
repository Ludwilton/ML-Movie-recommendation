# %%
import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler


# %%
ratings= pd.read_csv('./data/ratings.csv')
movies = pd.read_csv('./data/movies.csv')
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))



# %%
def filter_by_average_rating(movies_df, ratings_df, min_rating=2.5):
    print(f"Filtering movies with average rating >= {min_rating}...")
    
    movies_with_ratings = pd.merge(movies_df, ratings_df, on='movieId', how='inner')
    
    average_ratings = movies_with_ratings.groupby('movieId')['rating'].mean()
    
    popular_movies = average_ratings[average_ratings >= min_rating].index
    
    filtered_movies_with_ratings = movies_with_ratings[movies_with_ratings['movieId'].isin(popular_movies)]
    
    print(f"Kept {len(popular_movies)} movies with average rating >= {min_rating}")
    
    return filtered_movies_with_ratings

# %%

def filter_users_by_activity(ratings_df, min_percentile=0.85, max_percentile=0.97):

    print(f"Filtering users with activity between {min_percentile*100:.0f}th and {max_percentile*100:.0f}th percentiles...")
    
    # Count ratings per user
    user_rating_counts = ratings_df.groupby('userId')['rating'].count()
    print(f"Before filtering: {len(user_rating_counts)} users with rating statistics:")
    print(user_rating_counts.describe())
    
    # Get percentile thresholds
    min_threshold = user_rating_counts.quantile(min_percentile)
    max_threshold = user_rating_counts.quantile(max_percentile)
    
    # Filter users
    filtered_users = user_rating_counts[(user_rating_counts >= min_threshold) & 
                                       (user_rating_counts <= max_threshold)].index
    
    filtered_ratings = ratings_df[ratings_df['userId'].isin(filtered_users)]
    
    # Get statistics after filtering
    user_rating_counts_after = filtered_ratings.groupby('userId')['rating'].count()
    print(f"After filtering: {len(user_rating_counts_after)} users with rating statistics:")
    print(user_rating_counts_after.describe())
    
    return filtered_ratings

# %%

def filter_movies_by_popularity(ratings_df, min_ratings=20):

    print(f"Filtering movies with at least {min_ratings} ratings...")
    
    # Count ratings per movie
    movie_rating_counts = ratings_df.groupby('movieId').size()
    
    print(f"Before filtering: {len(movie_rating_counts)} movies")
    print(f"Movies with <{min_ratings} ratings: {(movie_rating_counts < min_ratings).sum()}")
    
    # Filter movies with sufficient ratings
    popular_movies = movie_rating_counts[movie_rating_counts >= min_ratings].index
    
    filtered_ratings = ratings_df[ratings_df['movieId'].isin(popular_movies)]
    
    print(f"After filtering: {len(popular_movies)} movies kept")
    
    return filtered_ratings

# %%
def select_diverse_users(ratings_df, max_users_per_movie=100, similarity_threshold=0.3):

    print(f"Selecting diverse users with max {max_users_per_movie} users per movie...")
    

    # Get initial statistics
    total_users = ratings_df['userId'].nunique()
    total_movies = ratings_df['movieId'].nunique()
    total_ratings = len(ratings_df)
    
    print(f"Initial dataset: {total_ratings} ratings from {total_users} users on {total_movies} movies")
    
    # Create a movie-to-users dictionary to track how many users rated each movie
    movie_to_users = {}
    for movie_id, group in ratings_df.groupby('movieId'):
        movie_to_users[movie_id] = set(group['userId'])
    
    # Create a user-to-movies dictionary
    user_to_movies = {}
    for user_id, group in ratings_df.groupby('userId'):
        user_to_movies[user_id] = set(group['movieId'])
    
    # Sort users by the number of ratings (to prioritize users with fewer ratings)
    user_rating_counts = ratings_df.groupby('userId').size()
    sorted_users = user_rating_counts.sort_values().index.tolist()
    
    # Shuffle the users to avoid bias toward specific user IDs
    np.random.shuffle(sorted_users)
    
    selected_users = set()
    movie_user_counts = {movie_id: 0 for movie_id in movie_to_users.keys()}
    
    # Process users in the sorted order
    for user_id in sorted_users:
        user_movies = user_to_movies[user_id]
        
        # Check if this user would exceed the max users for any movie
        exceeds_limit = False
        for movie_id in user_movies:
            if movie_user_counts[movie_id] >= max_users_per_movie:
                exceeds_limit = True
                break
        
        # Check similarity with already selected users
        too_similar = False
        if not exceeds_limit:
            for selected_user in selected_users:
                selected_user_movies = user_to_movies[selected_user]
                
                # Calculate Jaccard similarity between users
                intersection = len(user_movies.intersection(selected_user_movies))
                union = len(user_movies.union(selected_user_movies))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > similarity_threshold:
                        too_similar = True
                        break
        
        # Add user if they don't exceed limits and aren't too similar
        if not exceeds_limit and not too_similar:
            selected_users.add(user_id)
            
            # Update movie user counts
            for movie_id in user_movies:
                movie_user_counts[movie_id] += 1
    
    # Filter the ratings DataFrame to only include selected users
    filtered_ratings = ratings_df[ratings_df['userId'].isin(selected_users)]
    
    print(f"After filtering: {len(filtered_ratings)} ratings from {len(selected_users)} users")
    print(f"Retained {len(filtered_ratings)/total_ratings:.1%} of original ratings")
    
    return filtered_ratings

# %%
def create_filtered_dataset(movies_df, ratings_df, 
                           min_avg_rating=2.5,
                           user_min_percentile=0.85, 
                           user_max_percentile=0.97,
                           min_ratings_per_movie=20,
                           max_users_per_movie=100,
                           user_similarity_threshold=0.3,):

    print("Starting dataset filtering pipeline...")
    
    # Step 1: Apply diverse user selection first
    ratings_df = select_diverse_users(
        ratings_df, 
        max_users_per_movie=max_users_per_movie,
        similarity_threshold=user_similarity_threshold,
    )
    
    # Step 2: Filter by average rating
    filtered_df = filter_by_average_rating(
        movies_df, 
        ratings_df, 
        min_rating=min_avg_rating
    )
    
    # Step 3: Filter users by activity level
    filtered_df = filter_users_by_activity(
        filtered_df,
        min_percentile=user_min_percentile,
        max_percentile=user_max_percentile
    )
    
    # Step 4: Filter movies by popularity
    filtered_df = filter_movies_by_popularity(
        filtered_df,
        min_ratings=min_ratings_per_movie
    )
    
    # Drop unnecessary columns
    if 'timestamp' in filtered_df.columns:
        filtered_df.drop(columns=["timestamp"], inplace=True)
    if 'genres' in filtered_df.columns:
        filtered_df.drop(columns=["genres"], inplace=True)
    if 'title' in filtered_df.columns:
        filtered_df.drop(columns=["title"], inplace=True)
        
    print(f"Final dataset: {len(filtered_df)} ratings across {filtered_df['movieId'].nunique()} movies from {filtered_df['userId'].nunique()} users")
    
    return filtered_df

# %%
# Load data
ratings = pd.read_csv('./data/ratings.csv')
movies = pd.read_csv('./data/movies.csv')

# Create a filtered dataset with all steps
filtered_df = create_filtered_dataset(
    movies, 
    ratings,
    min_avg_rating=2.5,
    user_min_percentile=0.85,
    user_max_percentile=0.97,
    min_ratings_per_movie=50,
    max_users_per_movie=100,
    user_similarity_threshold=0.3
)

# Create user-movie matrix
user_movie_matrix = filtered_df.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.fillna(0)

# %%
def scale_ratings(matrix):

    data = matrix.copy().values
    rows = data.shape[0]
    
    # Scale each user's ratings (each row)
    for i in range(rows):
        # Find rated movies for this user
        nonzero_mask = data[i, :] > 0
        rated_indices = np.where(nonzero_mask)[0]
        
        # Get user's ratings and reshape for scaler
        user_ratings = data[i, rated_indices].reshape(-1, 1)
        
        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        scaled_ratings = scaler.fit_transform(user_ratings).flatten()
        
        # Update the matrix with scaled ratings
        data[i, rated_indices] = scaled_ratings
    
    # Convert back to DataFrame
    scaled_matrix = pd.DataFrame(data, index=matrix.index, columns=matrix.columns)
    return scaled_matrix

# %%
def prepare_user_ratings(ratings_file_path, user_movie_matrix_columns):

    user_data = pd.read_csv(ratings_file_path)
    user_data = user_data.dropna(subset=['movieId'])
    
    # Create user-movie ratings matrix with zero values - using float dtype
    user_ratings = pd.DataFrame(0.0, index=[0], columns=user_movie_matrix_columns, dtype=np.float64)
    
    # Fill with original ratings
    for _, row in user_data.iterrows():
        movie_id = row['movieId']
        if movie_id in user_ratings.columns:
            user_ratings.loc[0, movie_id] = float(row['Rating'])
    
    # Get indices of rated movies
    nonzero_mask = user_ratings.values > 0
    rated_indices = np.where(nonzero_mask[0])[0]
    
    # Get this user's ratings
    user_rating_values = user_ratings.iloc[0, rated_indices].values.reshape(-1, 1)
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    scaled_ratings = scaler.fit_transform(user_rating_values).flatten()
    
    # Update the user ratings with scaled values
    user_ratings.iloc[0, rated_indices] = scaled_ratings
    
    return user_data, user_ratings

# %%
def fit_model(
    user_movie_matrix_scaled,
    n_components=50,
    max_iter=200,
    init='random',
    solver='cd',
    tol=0.0001,

    sample_size=1.0
):

    print(f"\n--- Fitting model with {sample_size*100:.0f}% of users ---")
    
    # Sample users 
    if sample_size < 1.0:
        n_users = int(user_movie_matrix.shape[0] * sample_size)
        sampled_users = np.random.choice(user_movie_matrix.index, size=n_users, replace=False)
        training_matrix = user_movie_matrix.loc[sampled_users, :]
    else:
        training_matrix = user_movie_matrix
    
    print(f"Training matrix shape: {training_matrix.shape}")
    
    # Train the model
    start_time = time.time()
    nmf = NMF(
        n_components=n_components,
        max_iter=max_iter,
        verbose=0,
        init="nndsvd",
        #solver=solver,
        #tol=tol,
        #l1_ratio=l1_ratio
    )
    
    user_factors = nmf.fit_transform(training_matrix)
    item_factors = nmf.components_
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Iterations completed: {nmf.n_iter_}")
    print(f"Final error: {nmf.reconstruction_err_}")
    
    return nmf, training_matrix, training_time



# %%
def get_recommendations(
    nmf_model,
    training_matrix,
    my_ratings,
    my_data,
    n_recommendations=20,
    movie_id_to_title_map=None
):
    # Transform personal ratings into factor space
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
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    top_recommendations = recommendations[:n_recommendations]
    
    if movie_id_to_title_map:
        print("\nTop Recommendations:")
        for movie_id, predicted_rating in top_recommendations:
            # Convert from 0-1 scale back to 1-5 scale
            original_scale_rating = predicted_rating * 4 + 1
            
            movie_title = movie_id_to_title_map.get(movie_id, f"Unknown Movie (ID: {movie_id})")
            print(f"Movie: {movie_title}, Predicted Rating: {original_scale_rating:.2f}/5.00")
    
    return top_recommendations

# %%
user_movie_matrix_scaled = scale_ratings(user_movie_matrix)


# %%
nmf_model, training_matrix, _ = fit_model(
    user_movie_matrix_scaled,
    n_components=30,
    sample_size=1,
    max_iter=400
)

# %%
elliott_data, elliott_ratings = prepare_user_ratings(
    "data/loelliot_ratings_with_ids.csv", 
    user_movie_matrix_scaled.columns
)
ludde_data, ludde_ratings = prepare_user_ratings(
    "data/lddec_ratings_with_ids.csv", 
    user_movie_matrix_scaled.columns
)
charlie_data, charlie_ratings = prepare_user_ratings(
    "data/chaarll_ratings_with_ids.csv", 
    user_movie_matrix_scaled.columns
)
tilda_data, tilda_ratings = prepare_user_ratings(
    "data/tilda_h_ratings_with_ids.csv",
    user_movie_matrix_scaled.columns
)

# %%
recommendations1 = get_recommendations(
    nmf_model=nmf_model,
    training_matrix=training_matrix,
    my_ratings=ludde_ratings,
    my_data=ludde_data,
    n_recommendations=200,
    movie_id_to_title_map=movie_id_to_title
)


# %%
recommendations = get_recommendations(
    nmf_model=nmf_model,
    training_matrix=training_matrix,
    my_ratings=tilda_ratings,
    my_data=tilda_data,
    n_recommendations=200,
    movie_id_to_title_map=movie_id_to_title,
)

# %%

recommendations = get_recommendations(
    nmf_model=nmf_model,
    training_matrix=training_matrix,
    my_ratings=elliott_ratings,
    my_data=elliott_data,
    n_recommendations=200,
    movie_id_to_title_map=movie_id_to_title,
)

# %%
recommendations = get_recommendations(
    nmf_model=nmf_model,
    training_matrix=training_matrix,
    my_ratings=charlie_ratings,
    my_data=charlie_data,
    n_recommendations=200,
    movie_id_to_title_map=movie_id_to_title,
)


