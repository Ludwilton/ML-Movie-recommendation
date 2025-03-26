import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

base_url = "https://letterboxd.com/lddec/films/page/"


def fetch_average_user_rating(film_page_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    with webdriver.Chrome(options=options) as driver:
        driver.get(film_page_url)
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'section.ratings-histogram-chart span.average-rating a'))
            )
            
            avg_rating_tag = driver.find_element(By.CSS_SELECTOR, 'section.ratings-histogram-chart span.average-rating a')
            avg_rating_value = avg_rating_tag.text.strip()
            avg_rating_text = avg_rating_tag.get_attribute('data-original-title')
            
            avg_rating_number = avg_rating_value.split()[0]
            print(f"Average rating for {film_page_url}: {avg_rating_text} ({avg_rating_number})")
            return avg_rating_number, avg_rating_text
            
        except Exception as e:
            print(f"Average rating for {film_page_url}: No Rating (No Rating), Error: {e}")
            return "No Rating", "No Rating"




def scrape_letterboxd(url, single_film=False):
    response = requests.get(url)
    response.raise_for_status()  
    print(f"Fetched page: {url}")

    soup = BeautifulSoup(response.text, 'html.parser')
    

    films = soup.find_all('li', class_='poster-container')
    print(f"Found {len(films)} films on this page.")

    film_data = []
    
    for index, film in enumerate(films):
        if single_film and index >= 1:
            break


        film_div = film.find('div', class_='really-lazy-load')
        
        if film_div:

            film_slug = film_div.get('data-film-slug')
            film_link = film_div.get('data-target-link')

            if not film_slug or not film_link:
                print("Missing film slug or target link, skipping this film.")
                continue


            film_page_url = f"https://letterboxd.com{film_link}"
            print(f"Fetching genres and average rating for {film_slug}...")


            film_response = requests.get(film_page_url, timeout=10)
            film_response.raise_for_status()
            
            film_soup = BeautifulSoup(film_response.text, 'html.parser')


            genres = film_soup.find_all('a', class_='text-slug')
            film_genres = [genre.text.strip() for genre in genres if genre.has_attr('href') and 'genre' in genre['href']]
            

            avg_rating_value, avg_rating_text = fetch_average_user_rating(film_page_url)


            rating = film.find('p', class_='poster-viewingdata').find('span', class_='rating')
            film_rating = rating.text.strip() if rating else "No Rating"

            film_data.append({
                'name': film_slug,
                'rating': film_rating,
                'genres': ', '.join(film_genres),
                'avg_user_rating': avg_rating_value
            })

    return film_data



def save_to_csv(film_data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'rating', 'genres', 'avg_user_rating'])
        writer.writeheader()
        for film in film_data:
            writer.writerow(film)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    all_films = []
    
    option = input("Enter 'single' to scrape a single film or 'all' to scrape all films across pages: ").strip().lower()

    if option == 'single': # test
        url = f"{base_url}1/"
        films_info = scrape_letterboxd(url, single_film=True)
        all_films.extend(films_info)
    elif option == 'all':
        # Dynamically determine the number of pages
        response = requests.get(f"{base_url}1/")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        pagination = soup.find('div', class_='paginate-pages')
        if pagination:
            last_page_link = pagination.find_all('a')[-1]
            total_pages = int(last_page_link.text.strip())
        else:
            total_pages = 1  # Default to 1 if pagination is not found

        for page in range(1, total_pages + 1):
            url = f"{base_url}{page}/"
            films_info = scrape_letterboxd(url)
            all_films.extend(films_info)

    if all_films:
        save_to_csv(all_films, 'letterboxd_films.csv')
    else:
        print("No films found.")
