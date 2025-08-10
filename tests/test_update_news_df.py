import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Web_Scraper import update_news_df


def test_update_news_df_drops_duplicate_headlines(monkeypatch):
    existing_df = pd.DataFrame([
        {"headline": "Old Headline", "news_type": "market", "scrape_time": pd.Timestamp("2024-01-01")}
    ])

    mock_headlines = [
        {"headline": "Old Headline", "news_type": "market", "scrape_time": pd.Timestamp("2024-01-02")},
        {"headline": "New Headline", "news_type": "market", "scrape_time": pd.Timestamp("2024-01-02")},
        {"headline": "New Headline", "news_type": "market", "scrape_time": pd.Timestamp("2024-01-02")},
    ]

    def mock_scrape(url, news_type, headless=True):
        return mock_headlines

    monkeypatch.setattr("Web_Scraper.scrape_finviz_news", mock_scrape)

    result_df = update_news_df(existing_df)

    assert len(result_df) == 2
    assert (result_df["headline"] == "Old Headline").sum() == 1
    assert (result_df["headline"] == "New Headline").sum() == 1
