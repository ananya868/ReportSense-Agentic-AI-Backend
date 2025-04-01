import os, json 
import re 
import tiktoken
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio

from pydantic import BaseModel, Field
from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI
from groq import Groq
import instructor

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

    def clean_pages(self, pages: list[str], max_tokens: int = 1200):
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
    
    def llm(self, cleaned_page: str, provider: str = "openai"):
        """
        Extracts the prices from the cleaned pages using the Groq API.
        """
        assert provider in ["openai", "groq"], "Invalid provider. Choose either 'openai' or 'groq'."
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            try:
                client = OpenAI(api_key=api_key)
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {e}")
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            try:
                groq_client = Groq(api_key=api_key)
                client = instructor.from_groq(groq_client)
            except Exception as e:
                raise ValueError(f"Failed to initialize Groq client: {e}")

        prompt_message = f"""
            Given below is a text fetched from a pharmacy website. Please extract the prices of the medicine: {self.medicine_name} from the text.
            Also the number of tablets (if given). 
            Text: {cleaned_page}
            Your output should be dictionary format with the following fields:
            - name: The name of the medicine
            - price: The price of the medicine
            - quantity: The number of tablets (if given)
        - Only output in the specified format. If the information is not available, return "NA" for the respective field.

        """ 
        class ResponseModel(BaseModel):
            name: str = Field(..., description="The name of the medicine")
            price: str = Field(... , description="The price of the medicine")
            quantity: str = Field(... , description="The number of tablets (if given)")

        # LLM 
        if provider == "openai":
            completion = client.beta.chat.completions.parse(
                model = "gpt-4o-mini",
                messages = [
                    { 
                        "role": "user",
                        "content": prompt_message
                    }
                ],
                response_format = ResponseModel
            )
            response = completion.choices[0].message.parsed
        elif provider == "groq":
            completion = client.chat.completions.create(
                model = "llama-3.3-70b-versatile",
                response_model = ResponseModel,
                messages = [
                    { 
                        "role": "user",
                        "content": prompt_message
                    }
                ],
            )
            response = completion
        return response

    def get_prices(self, urls: list[str], cleaned_pages: list[str], provider: str = "openai"):
        """
        Extracts the prices from the cleaned pages using regex.
        """
        prices = []
        # Convert the cleaned pages to json format

        for url, page in tqdm(zip(urls, cleaned_pages)):
            data = self.llm(cleaned_page = page, provider = provider)
            # Convert the response to a dictionary
            data_dict = {
                "name": data.name,
                "price": data.price,
                "quantity": data.quantity,
                "url": url
            }
            # Append the data to the list
            prices.append(data_dict)

        return prices




# Example usage 
if __name__ == "__main__":
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import time

    for i in ['ibuprofen', 'paracetamol']:
        print("Fetching medicine prices...")
        medicine_name = i
        fetcher = FetchMedicinePrices(medicine_name)
        urls = fetcher.fetch_links()
        
        print("Scraping the websites...")
        # Scrapper 
        pages = asyncio.run(fetcher.fetch_prices(urls))
        
        print("Cleaning the pages...")
        # Cleaning
        cleaned_pages = fetcher.clean_pages(pages)

        print("Extracting prices...")

        s = time.time()
        prices = fetcher.get_prices(urls = urls, cleaned_pages = cleaned_pages, provider = "openai")
        e = time.time()
        print(f"Time taken: {e-s} seconds") 
        print(prices)
        print(" -- - - -- - -- - -- - -")