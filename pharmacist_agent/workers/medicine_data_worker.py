"""
This module provides a class for fetching and processing medication information from the web. 
"""

import os, json
import asyncio
from dotenv import load_dotenv
from typing import Any 

from workers.schema import MedicationDetails

from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI

load_dotenv()


class FetchMedicineData:
    """A class for fetching and processing medication information from the web."""
    
    def __init__(self, medicine_name: str) -> None:
        """
        Initialize the FetchMedicineData instance.
        
        Args:
            medicine_name (str): The name of the medicine to search for.
        """
        self.medicine_name = medicine_name
    
    def search_web(self, num_res: int = 3) -> list[str]:
        """
        Search the web for information about the medicine.
        
        Args:
            num_res (int, optional): Number of search results to return. Defaults to 1.
            
        Returns:
            list[str]: A list of URLs from the search results.
        """
        res = [result for result in search(
            f"{self.medicine_name} drugs.com", num_results=num_res 
        )]
        # Check if "https or http in the urls list", pick up the first one with https or http
        for i in res:
            if "https" in i or "http" in i:
                res = [i]
                break
        return res
    
    async def fetch_webpage(self, url: str) -> list[str]:
        """
        Asynchronously fetch content from the provided URL(s).
        
        Args:
            url (str): URL or list of URLs to fetch content from.
            
        Returns:
            list[str]: A list of markdown-formatted content from the fetched webpages.
            
        Note:
            If a URL fails to fetch, an error message will be printed to the console.
        """
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Don't use cached results
            exclude_external_links=True,  # Don't follow external links
            excluded_tags=["form", "header", "footer", "nav"],  # Exclude these HTML elements
            stream=False  # Don't stream results
        )
        browser_conf = BrowserConfig(headless=True)  # Run browser in headless mode
        
        async with AsyncWebCrawler(browser_config=browser_conf) as crawler:
            results = await crawler.arun_many(url, config=run_conf)
            fetched_content = []
            
            for i, res in enumerate(results):
                if res.success:
                    fetched_content.append(res.markdown)
                else:
                    print(f"[ERROR] Failed to fetch {url[i]}")
            return fetched_content
    
    def generate_data_points(self, prompt: str, sys_prompt: str) -> Any:
        """
        Process the fetched content using OpenAI's API to extract structured medication information.
        
        Args:
            prompt (str): The user prompt to send to the OpenAI API.
            sys_prompt (str): The system prompt to send to the OpenAI API.
            
        Returns:
            dict: A dictionary containing structured medication information parsed from the API response.
        """
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format=MedicationDetails
        )
        return completion.choices[0].message.parsed





