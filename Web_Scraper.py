import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time

def scrape_finviz_news(url, news_type, headless=True):
    """
    Scrapes headlines from the given Finviz URL and labels them with the specified news_type.
    Returns a list of dictionaries containing headline, news_type, and scrape_time.
    """
    options = Options()
    if headless:
        options.add_argument("--headless")  # run browser in headless mode if desired
    # Set a user-agent to mimic a real browser
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/115.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)

    headlines_list = []
    try:
        driver.get(url)
        # Give the page some time to load completely
        time.sleep(3)

        wait = WebDriverWait(driver, 15)
        headline_elements = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.nn-tab-link"))
        )

        for elem in headline_elements:
            text = elem.text.strip()
            if text:  # Ensure non-empty headlines
                headlines_list.append({
                    'headline': text,
                    'news_type': news_type,
                    'scrape_time': datetime.now()
                })
        return headlines_list
    except Exception as e:
        print(f"Error scraping {news_type} news from {url}: {e}")
        # Print a snippet of the page source for debugging purposes
        print("Page source snippet:", driver.page_source[:500])
        return []
    finally:
        driver.quit()


def update_news_df(existing_df):
    """
    Scrapes headlines from various Finviz news pages (market, stock, ETF, crypto),
    labels them, and appends new (non-duplicate) headlines to the existing DataFrame.
    """
    urls = {
        'market': 'https://finviz.com/news.ashx',
        'stock': 'https://finviz.com/news.ashx?v=3',
        'etf': 'https://finviz.com/news.ashx?v=4',
        'crypto': 'https://finviz.com/news.ashx?v=5'
    }

    all_news = []
    for news_type, url in urls.items():
        news = scrape_finviz_news(url, news_type, headless=True)  # Set headless=False for troubleshooting
        all_news.extend(news)

    new_df = pd.DataFrame(all_news)

    if existing_df is not None and not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates based on the headline text
        combined_df.drop_duplicates(subset='headline', keep='first', inplace=True)
    else:
        combined_df = new_df

    return combined_df


def scheduled_job():
    '''
    load existing data, scrape new headlines and retires up to max attempts if the fetch failed and new data returned
    '''
    try:
        existing_df = pd.read_csv('finviz_news.csv', parse_dates=['scrape_time'])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing_df = pd.DataFrame(columns=['headline', 'news_type', 'scrape_time'])

    max_attempts = 2
    attempt = 0
    success = False
    updated_df = None

    while attempt < max_attempts and not success:
        print(f'Scraping at {datetime.now()} (attempt {attempt + 1} of {max_attempts})')
        updated_df = update_news_df(existing_df)
        if not updated_df.empty and len(updated_df) > len(existing_df):
            success = True
        else:
            print('Scraping failed, retrying...')
        attempt += 1

    if success:
        print(f'Scraping successful, saving to CSV... adding {len(updated_df) - len(existing_df)} new headlines to existing data')
        updated_df.to_csv('finviz_news.csv', index=False)

    else:
        print('Scraping failed, no new data returned.')

if __name__ == "__main__":
    # Attempt to load an existing CSV file; if not, start with an empty DataFrame

    scheduler = BlockingScheduler()

    scheduler.add_job(scheduled_job, 'interval', minutes=10)

    print(f'Starting scheduler... at time {datetime.now()}')

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print('Stopping scheduler...')
