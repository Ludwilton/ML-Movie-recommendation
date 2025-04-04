#%%
import numpy as np
import pandas as pd
import joblib 
import re



def load_recommendation_model():
    model = joblib.load("model_files/nmf_model.joblib")
    movie_ids = joblib.load("model_files/movie_ids.joblib") # we need this to map the user ratings to the model's movie IDs
    return model, movie_ids


def load_user_ratings(path, movie_ids):
    user_data = pd.read_csv(path)
    user_ratings = pd.DataFrame(0.0, index=[0], columns=movie_ids)
    
    for _, row in user_data.iterrows():
        movie_id = row["movieId"]
        if movie_id in movie_ids:
            user_ratings.loc[0, movie_id] = float(row["Rating"])
    
    ratings = user_ratings.values[0]
    rated_mask = ratings > 0
    nonzero_ratings = ratings[rated_mask]
    
    
    if len(nonzero_ratings) > 1 and np.std(nonzero_ratings) > 0: # z-zcore
        mean = np.mean(nonzero_ratings)
        std = np.std(nonzero_ratings)
        
        normalized = (nonzero_ratings - mean) / std
        normalized = np.clip(normalized, -2.5, 2.5)  
        normalized = (normalized + 2.5) / 5.0
        
        user_ratings.values[0, rated_mask] = normalized
    
    return user_ratings

def get_recommendations(nmf_model, movie_ids, user_ratings, n_recommendations=100):
    user_factors = nmf_model.transform(user_ratings.values)

    predicted_ratings = user_factors.dot(nmf_model.components_)[0]

    rated_mask = user_ratings.values[0] > 0
    scores = predicted_ratings.copy()
    scores[rated_mask] = -1
    
    top_indices = np.argsort(-scores)[:n_recommendations]
    recommendations = [movie_ids[idx] for idx in top_indices]
    
    return recommendations


def display_recommendations(movie_ids, movie_id_to_title_map):
    print("\nTop Recommendations:")
    for i, movie_id in enumerate(movie_ids, 1):
        title = movie_id_to_title_map[movie_id]
        print(f"{i}. {title}")


def find_best_movie_match(title, movies_df):
    title_lower = title.lower().strip()
    
    exact_matches = movies_df[movies_df['title'].str.lower() == title_lower]
    if not exact_matches.empty:
        return exact_matches.iloc[0]
    
    title_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', title_lower)
    
    for _, movie in movies_df.iterrows():
        movie_title = movie['title'].lower()
        movie_title_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_title)
        
        if movie_title_no_year == title_no_year:
            return movie
    
    substring_matches = movies_df[movies_df['title'].str.lower().str.contains(title_lower, regex=False)]
    if not substring_matches.empty:
        return substring_matches.iloc[0]
    
    potential_matches = []
    title_words = set(title_no_year.split())
    
    for _, movie in movies_df.iterrows():
        movie_title = movie['title'].lower()
        movie_title_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_title)
        movie_words = set(movie_title_no_year.split())
        
        common_words = title_words.intersection(movie_words)
        if common_words:
            word_score = len(common_words) / max(len(title_words), len(movie_words))
            length_penalty = abs(len(movie_title_no_year) - len(title_no_year)) / 100
            score = word_score - length_penalty
            
            potential_matches.append((score, movie))
    
    if potential_matches:
        potential_matches.sort(key=lambda x: x[0], reverse=True)
        return potential_matches[0][1]
    
    return None



def recommend_from_title(title, model, movie_ids, movies_df, n_recommendations=10):
    match = find_best_movie_match(title, movies_df)
    
    if match is None:
        print(f"No movies found matching '{title}'")
        return []
    

    movie_id = match['movieId']
    movie_title = match['title']
    print(f"Using: {movie_title}")
    
    user_ratings = pd.DataFrame(0.0, index=[0], columns=movie_ids)
    
    if movie_id in movie_ids:
        user_ratings.loc[0, movie_id] = 5.0
        recommendations = get_recommendations(model, movie_ids, user_ratings, n_recommendations)
        
        print(f"\nIf you like {movie_title}, you might also enjoy:")
        
        return recommendations
    else:
        print("Movie not in model database")
        return []



if __name__ == "__main__":
    model, movie_ids = load_recommendation_model()
    movies_df = pd.read_csv("data/movies.csv")
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # based on user data taken from letterboxd
    user_ratings = load_user_ratings("data/zorrodor_ratings_with_ids.csv", movie_ids)
    recommended_movie_ids = get_recommendations(model, movie_ids, user_ratings)
    display_recommendations(recommended_movie_ids, movie_id_to_title)

    # get recommendations based on a single movie
    # movie_title = "Pearl (2022)"
    # similar_movies = recommend_from_title(movie_title, model, movie_ids, movies_df,20)
    # display_recommendations(similar_movies, movie_id_to_title)
# 
# %%
