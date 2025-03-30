import os, json
import asyncio

from workers.schema import MedicationDetails

from googlesearch import search 
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


class FetchMedicineData:
    """
    A class for fetching and processing medication information from the web.
    This class uses Google search to find relevant URLs, web crawling to fetch content,
    and OpenAI's API to extract structured medication information from the fetched content.
    
    Attributes:
        medicine_name (str): The name of the medicine to search for.
    """
    
    def __init__(self, medicine_name: str):
        """
        Initialize the FetchMedicineData instance.
        
        Args:
            medicine_name (str): The name of the medicine to search for.
        """
        self.medicine_name = medicine_name
    
    def search_web(self, num_res: int = 1) -> list[str]:
        """
        Search the web for information about the medicine.
        
        Uses Google search to find URLs containing information about the medicine,
        primarily targeting drugs.com for reliable pharmaceutical information.
        
        Args:
            num_res (int, optional): Number of search results to return. Defaults to 1.
            
        Returns:
            list[str]: A list of URLs from the search results.
        """
        return [result for result in search(
            f"{self.medicine_name} drugs.com", num_results=num_res, unique=False, advanced=False
        )]
    
    async def fetch_webpage(self, url: str) -> list[str]:
        """
        Asynchronously fetch content from the provided URL(s).
        
        Uses AsyncWebCrawler to scrape web content, with configuration to exclude
        certain HTML elements and convert the content to markdown format.
        
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
    
    def generate_data_points(self, prompt: str, sys_prompt: str) -> dict:
        """
        Process the fetched content using OpenAI's API to extract structured medication information.
        
        This method sends the fetched content to OpenAI's API along with system and user prompts,
        and expects a response that can be parsed into a MedicationDetails object.
        
        Args:
            prompt (str): The user prompt to send to the OpenAI API.
            sys_prompt (str): The system prompt to send to the OpenAI API.
            
        Returns:
            dict: A dictionary containing structured medication information parsed from
                  the OpenAI API response as a MedicationDetails object.
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





# Example Usage
if __name__=="__main__":
    fetcher = FetchMedicineData(medicine_name="Levetiracetam")
    print(fetcher.medicine_name)
    search_results = fetcher.search_web()
    print(search_results)
    print("Fetching Webpage...")
    page = asyncio.run(fetcher.fetch_webpage(search_results))
    context = page[0]
    # Define the prompt and system prompt 
    prompt = f"""Analyze the provided webpage content and extract structured details about the medication using the following fields:
        **Important Guidelines**:
        1. If any information is missing in the provided context, return `"missing"` as its value.
        2. Do NOT generate or assume any information not explicitly found in the context.
        3. Return the extracted details in JSON format.
        Now, process the following webpage content and generate the structured output:
        \n\n{context}
    """
    sys_prompt = """You are a highly intelligent medical assistant designed to extract structured information about medications from a given webpage. 
        Your goal is to analyze the provided context carefully and fill in the relevant fields. 
        If a particular piece of information is not found, return "missing" as its value instead of leaving it blank. 

    #     Ensure accuracy while extracting details and avoid making assumptions. Only use information explicitly stated in the context.
    # """
    print("Generating Data Points...")
    # Generate data points
    data_points = fetcher.generate_data_points(context, prompt, sys_prompt)
    # Convert to json/dict 
    data_dict = data_points.dict()
    # dump json 
    with open("medicine_data.json", "w") as f: 
        json.dump(data_dict, f, indent=4)
