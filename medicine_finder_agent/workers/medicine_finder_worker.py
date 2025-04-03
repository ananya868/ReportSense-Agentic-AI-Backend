"""
This script fetches the prices of a given medicine from various pharmacy websites using web scraping and LLMs.
It uses the `crawl4ai` library for web crawling and the `instructor` library for LLM processing. 
"""

import os, json 
import re 
import tiktoken
from tqdm import tqdm
import asyncio
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Any

from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI
from groq import Groq
import instructor

load_dotenv()


class FetchMedicinePrices:
    """
    A class to fetch the prices of a medicine from various pharmacy websites.
    """

    def __init__(self, medicine_name: str) -> None:
        """
        Initializes the FetchMedicinePrices object with the name of the medicine.

        Args:
            medicine_name (str): The name of the medicine to fetch prices for.
        """
        self.medicine_name = medicine_name

    def fetch_links(self) -> list[str]:
        """ 
        Fetches the URLs of the websites that contain the prices of the medicine.

        Args: 
            medicine_name (str): The name of the medicine. 

        Returns:
            list[str]: A list of URLs containing the prices of the medicine.
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

    async def fetch_prices(self, urls: list[str]) -> list[str]:
        """
        Fetches the prices of the medicine from the given URLs.

        Args:
            urls (list[str]): A list of URLs to fetch prices from.

        Returns:
            list[str]: A list of fetched pages.
        """
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  
            exclude_external_links=True, 
            excluded_tags=["header", "footer", "nav"],
            stream=False  
        )
        browser_conf = BrowserConfig(headless=True)

        # Run Crawler 
        async with AsyncWebCrawler(browser_config=browser_conf) as crawler:
            results = await crawler.arun_many(urls, config=run_conf)
            fetched_pages = []

            for i, res in enumerate(results):
                if res.success:
                    fetched_pages.append(res.markdown)
                else: 
                    print(f"[ERROR] Failed to fetch {urls[i]}")
                
            return fetched_pages

    def clean_pages(self, pages: list[str], max_tokens: int = 1200) -> list[str]:
        """
        Cleans the fetched pages to extract the prices.

        Args:
            pages (list[str]): A list of fetched pages.
            max_tokens (int, optional): The maximum number of tokens allowed in the cleaned pages. Defaults to 1200.

        Returns:
            list[str]: A list of cleaned pages.
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
        return cleaned_pages
    
    def llm(self, cleaned_page: str, provider: str = "openai") -> Any:
        """
        Extracts the prices from the cleaned pages using the Groq API.

        Args:   
            cleaned_page (str): The cleaned page to extract prices from.
            provider (str, optional): The provider to use for LLM processing. Defaults to "openai".
        
        Returns:
            Any: The extracted prices.
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

        # Prompt message
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
        # Schema
        class ResponseModel(BaseModel):
            name: str = Field(..., description="The name of the medicine")
            price: str = Field(... , description="The price of the medicine")
            quantity: str = Field(... , description="The number of tablets (if given)")

        # LLM Run 
        try: 
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
        except Exception as e:
            raise ValueError(f"Failed to process LLM request: {e}")
        return response

    def get_prices(self, urls: list[str], cleaned_pages: list[str], provider: str = "openai") -> list[dict[str, str]]:
        """
        Process the cleaned pages using LLM to extract prices.

        Args:
            urls (list[str]): A list of URLs.
            cleaned_pages (list[str]): A list of cleaned pages.
            provider (str, optional): The provider to use for LLM processing. Defaults to "openai".
        
        Returns:
            list[dict[str, str]]: A list of dictionaries containing the extracted prices.
        """
        prices = []
        # Iterate through each cleaned page and extract prices using LLM
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


