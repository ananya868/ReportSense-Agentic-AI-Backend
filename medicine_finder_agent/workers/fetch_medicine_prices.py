import os, json 
from datetime import datetime

from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


class FetchMedicinePrices:
    """
    """ 

    def __init__(self, medicine_name: str):
        """
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




# Example usage 
if __name__ == "__main__":
    import asyncio

    medicine_name = "levipil 500"
    fetcher = FetchMedicinePrices(medicine_name)
    urls = fetcher.fetch_links()
    
    # Scrapper 
    pages = asyncio.run(fetcher.fetch_prices(urls))
    print(pages[3]) # Print the content of the first fetched page