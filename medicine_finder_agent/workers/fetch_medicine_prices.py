import os, json 
import re 
import tiktoken
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio

from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI
from groq import Groq

from dotenv import load_dotenv

load_dotenv()


class FetchMedicinePrices:
    """
    A class to fetch the prices of a medicine from various pharmacy websites.
    """

    def __init__(self, medicine_name: str):
        """
        Initializes the FetchMedicinePrices object with the name of the medicine.
        """
        self.medicine_name = medicine_name

    def fetch_links(self):
        """ 
        Fetches the URLs of the websites that contain the prices of the medicine.
        """
        # List of pharmacy domains to search
        pharmacy_domains = [
            "practo.com",
            "netmeds.com",
            "pharmeasy.in",
            "apollopharmacy.in",
            "medkart.in"
        ]
        
        urls = []
        
        # Iterate through each domain and search for the medicine
        for domain in pharmacy_domains:
            search_query = f"{self.medicine_name} {domain} price"
            for result in search(search_query, num_results=5, unique=False, advanced=False):
                if domain in result:
                    urls.append(result)
                    break
        
        return urls

    async def fetch_prices(self, urls: list[str]):
        """
        Fetches the prices of the medicine from the given URLs.
        """
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Don't use cached results # Don't follow internal links
            exclude_external_links=True,  # Don't follow external links
            excluded_tags=["header", "footer", "nav"],  # Exclude these HTML elements
            stream=False  
        )
        browser_conf = BrowserConfig(headless=True)

        async with AsyncWebCrawler(browser_config=browser_conf) as crawler:
            results = await crawler.arun_many(urls, config=run_conf)
            fetched_pages = []

            for i, res in enumerate(results):
                if res.success:
                    fetched_pages.append(res.markdown)
                else: 
                    print(f"[ERROR] Failed to fetch {urls[i]}")
                
            return fetched_pages

    def clean_pages(self, pages: list[str], max_tokens: int = 2000):
        """
        Cleans the fetched pages to extract the prices.
        """
        cleaned_pages = []

        for page in pages: 
            # Remove URLs 
            text = re.sub(r'http[s]?://\S+', '', page)
            # Remove non-alphanumeric characters (excluding spaces and basic punctuation)
            text = re.sub(r'[^\w\s.,!?]', '', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned_pages.append(text)

        # Maintain Token size for LLM 
        encoding = tiktoken.get_encoding("cl100k_base")
        result = []

        for page in cleaned_pages:
            tokens = encoding.encode(page)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                page = encoding.decode(tokens)
            result.append(page)

        cleaned_pages = result

        # print total number of tokens in each page
        for i, page in enumerate(cleaned_pages):
            print(f"Page {i+1} has {len(encoding.encode(page))} tokens")
        return cleaned_pages
    
    def llm(self, cleaned_page: str, api_key: str = os.getenv("GROQ_API_KEY")):
        """
        Extracts the prices from the cleaned pages using the Groq API.
        """
        client = Groq(api_key=api_key)

        prompt_message = f"""
            Given below is a text fetched from a pharmacy website. Please extract the prices of the medicine: {self.medicine_name} from the text.
            Also the number of tablets (if given). 
            Text: {cleaned_page}
            Your output should be dictionary format with the following fields:
            - name: The name of the medicine
            - price: The price of the medicine
            - quantity: The number of tablets (if given)
        - Only output the dictionary, no extra text! 

        """ 

        completion = client.chat.completions.create(
            messages = [
                { 
                    "role": "user",
                    "content": prompt_message
                }
            ], 
            model = "llama-3.3-70b-versatile"
        )

        return completion.choices[0].message.content

    def get_prices(self, cleaned_pages: list[str]):
        """
        Extracts the prices from the cleaned pages using regex.
        """
        prices = []
        # Convert the cleaned pages to json format

        for page in tqdm(cleaned_pages):
            info = self.llm(cleaned_page=page)
            info = json.loads(info)
            prices.append(info)

        return prices

# Example usage 
if __name__ == "__main__":
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import time

    print("Fetching medicine prices...")
    medicine_name = "frisium 10 mg"
    fetcher = FetchMedicinePrices(medicine_name)
    urls = fetcher.fetch_links()
    
    print("Scraping the websites...")
    # Scrapper 
    pages = asyncio.run(fetcher.fetch_prices(urls))
    
    print("Cleaning the pages...")
    # Cleaning
    cleaned_pages = fetcher.clean_pages(pages)

    print("Extracting prices...")
    # # prices 
    # prices = fetcher.get_prices(cleaned_pages)
    # print(prices)
    s = time.time()
    prices = fetcher.get_prices(cleaned_pages)
    e = time.time()
    print(f"Time taken: {e-s} seconds")
    print(type(prices[0])) 
    print(prices[0])