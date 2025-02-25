# Thematic Asset Allocation Pipeline

![image](https://github.com/user-attachments/assets/8c329857-a0f4-421f-bb9d-e42c38502b51)
---

# Thematic Portfolio & News 


This project leverages NLP techniques to analyze financial news headlines, extract sentiment, classify sectors via zero-shot learning, and identify target companies using NER. It then uses these signals—along with real-time market data—to construct a thematic portfolio using optimization methods.

## Features

- **Sentiment Analysis:**  
  Fine-tuned BERT model to predict headline sentiment.

- **Sector Classification:**  
  Zero-shot classification using Facebook's BART-large-MNLI to assign sector labels.

- **Entity Extraction:**  
  NER pipeline extracts organizations (candidate tickers) from headlines.

- **Theme Scoring:**  
  Keyword-based scoring for target themes (e.g., Cloud Computing, ESG).

- **Thematic Portfolio Optimization:**  
  Portfolio weights are optimized via CVXPY based on aggregated theme scores and real-time market data (retrieved using yfinance).

- **Data Visualization:**  
  Plots diverging sentiment charts for sectors and top securities.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/thematic-portfolio-news.git
   cd thematic-portfolio-news
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include packages like:
   - pandas
   - numpy
   - torch
   - transformers
   - datasets
   - yfinance
   - cvxpy
   - apscheduler
   - selenium
   - matplotlib
   - seaborn
   - tqdm

## Usage

### Web Scraper & News Aggregation

- The web scraper uses Selenium and APScheduler to periodically fetch headlines from Finviz.
- Run the scraper:
  ```bash
  python web_scraper.py
  ```

### NLP Analysis & Portfolio Construction

- The NLP pipeline extracts sentiment, sector, and target companies from headlines.
- The thematic portfolio is built using aggregated theme scores and real-time market data.
- To run the analysis and portfolio optimization:
  ```bash
  python portfolio_analysis.py
  ```

### Data Visualization

- The project includes notebooks/scripts to generate diverging sentiment visualizations for sectors and securities.
- For example, open and run the Jupyter Notebook:
  ```bash
  jupyter notebook visualization.ipynb
  ```

## Project Structure

```
├── README.md
├── requirements.txt
├── web_scraper.py         # Web scraper using Selenium & APScheduler
├── portfolio_analysis.py  # NLP processing and portfolio optimization
├── visualization.ipynb    # Jupyter Notebook for data visualization
├── models/                # Directory to store fine-tuned models
└── data/                  # Directory for scraped/aggregated data (e.g., CSV files)
```

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/yourusername/thematic-portfolio-news/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


