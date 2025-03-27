import pandas as pd
import requests
import re
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import os

def letterboxd_to_movielens(input_file):
    """
    Convert Letterboxd ratings to MovieLens format with only Rating and movieId columns.
    
    Parameters:
    -----------
    input_file : str
        Path to the Letterboxd ratings CSV file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with only Rating and movieId columns
    """
    # Generate output filename
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_path = f"data/{input_filename}_with_ids.csv"
    
    # Load data
    df = pd.read_csv(input_file)
    links_df = pd.read_csv('data/links.csv')
    
    #  extract IMDb IDs
    def get_imdb_id(letterboxd_uri):
        try:
            resp = requests.get(letterboxd_uri, allow_redirects=True)
            if resp.status_code != 200:
                return letterboxd_uri, None

            re_match = re.findall(r'href=".+title/(tt\d+)/maindetails"', resp.text)
            if not re_match:
                return letterboxd_uri, None

            return letterboxd_uri, re_match[0]
        except Exception as e:
            print(f"Error fetching {letterboxd_uri}: {e}")
            return letterboxd_uri, None
    
    print("Looking up IMDb IDs from Letterboxd...")

    
    letterboxd_uris = df['Letterboxd URI'].tolist()
    uri_to_imdb = {}
    max_workers = multiprocessing.cpu_count()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_imdb_id, uri): uri for uri in letterboxd_uris}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            uri, imdb_id = future.result()
            uri_to_imdb[uri] = imdb_id
    
    df['imdb_id'] = df['Letterboxd URI'].map(uri_to_imdb)
    
    # Convert IMDb IDs to match MovieLens format
    df['imdb_id_no_prefix'] = df['imdb_id'].apply(
        lambda x: int(x[2:]) if pd.notna(x) and isinstance(x, str) and x.startswith('tt') else None
    )
    
    # Match with MovieLens IDs
    matched_df = df.merge(
        links_df[['movieId', 'imdbId']], 
        left_on='imdb_id_no_prefix',
        right_on='imdbId', 
        how='left'
    )
    

    result_df = matched_df[['Rating', 'movieId']].dropna(subset=['movieId'])

    result_df['movieId'] = result_df['movieId'].astype(int)
    
    result_df.to_csv(output_path, index=False)
    
    print(f"Successfully matched {len(result_df)} out of {len(df)} movies ({len(result_df)/len(df)*100:.1f}%)")
    print(f"Saved to {output_path}")
    
    return result_df

if __name__ == "__main__":
    letterboxd_to_movielens('clar_ratings.csv')