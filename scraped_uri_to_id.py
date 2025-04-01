import requests
import pandas as pd
import time
from tqdm import tqdm
import re
import concurrent.futures
import multiprocessing

# thanks to https://github.com/TobiasPankner/Letterboxd-to-IMDb 

def get_imdb_id(letterboxd_uri):
    try:
        resp = requests.get(letterboxd_uri)
        if resp.status_code != 200:
            return letterboxd_uri, None

        # Extract the IMDb ID
        re_match = re.findall(r'href=".+title/(tt\d+)/maindetails"', resp.text)
        if not re_match:
            return letterboxd_uri, None

        return letterboxd_uri, re_match[0]
    except Exception as e:
        print(f"Error fetching {letterboxd_uri}: {e}")
        return letterboxd_uri, None

# Load data
letterboxd_df = pd.read_csv('liv_ratings.csv')
links_df = pd.read_csv('data/links.csv')

# Create a list of URIs to process
letterboxd_uris = [f"https://letterboxd.com/film/{name}/" for name in letterboxd_df['name']]

# Process in parallel using ThreadPoolExecutor
print("Looking up IMDb IDs from Letterboxd (in parallel)...")
start_time = time.time()

# Create a dictionary to store result
uri_to_imdb = {}

max_workers = multiprocessing.cpu_count()

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(get_imdb_id, uri): uri for uri in letterboxd_uris}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        uri, imdb_id = future.result()
        uri_to_imdb[uri] = imdb_id

# Map results back to the dataframe
letterboxd_df['imdb_id'] = [uri_to_imdb.get(f"https://letterboxd.com/film/{name}/") for name in letterboxd_df['name']]

end_time = time.time()
print(f"Letterboxd lookup completed in {end_time - start_time:.2f} seconds")

# Remove 'tt' prefix from IMDb IDs and convert to integer to match links.csv format
letterboxd_df['imdb_id_no_prefix'] = letterboxd_df['imdb_id'].apply(
    lambda x: int(x[2:]) if pd.notna(x) and x.startswith('tt') else None
)



print("Matching with MovieLens IDs...")
matched_df = letterboxd_df.merge(
    links_df[['movieId', 'imdbId']], 
    left_on='imdb_id_no_prefix',
    right_on='imdbId', 
    how='left'
)

# Add only the movieId column to the original dataframe
letterboxd_df['movieId'] = matched_df['movieId']

# Ensure the Rating column has the correct capitalization
if 'rating' in letterboxd_df.columns and 'Rating' not in letterboxd_df.columns:
    letterboxd_df['Rating'] = letterboxd_df['rating']
elif 'Rating' not in letterboxd_df.columns:
    print("Warning: No 'Rating' column found. Check your input file column names.")
    # If there's no rating column at all, create one with default values
    letterboxd_df['Rating'] = 5.0  # Default rating, change if needed

# Save only the Rating and movieId columns and remove any rows with NaN in either column
result_df = letterboxd_df[['Rating', 'movieId']].dropna()

# Save to CSV
result_df.to_csv('letterboxd_films_with_ids.csv', index=False)

# Print detailed statistics
matched = letterboxd_df['movieId'].notnull().sum()
valid_ratings = letterboxd_df['Rating'].notnull().sum()
complete_rows = len(result_df)

print(f"Successfully matched {matched} out of {len(letterboxd_df)} movies ({matched/len(letterboxd_df)*100:.1f}%)")
print(f"Valid ratings: {valid_ratings} out of {len(letterboxd_df)} movies ({valid_ratings/len(letterboxd_df)*100:.1f}%)")
print(f"Saved {complete_rows} movies with both valid ratings and IDs to 'letterboxd_films_with_ids.csv'")