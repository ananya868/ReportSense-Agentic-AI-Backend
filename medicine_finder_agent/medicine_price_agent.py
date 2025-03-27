from workers.fetch_medicine_prices import FetchMedicinePrices
from agent_models.medicine_price_models import MedicinePriceRequest, MedicinePriceResponse

from uagents import Agent, Context
import asyncio
import json
import os, time

import nest_asyncio
nest_asyncio.apply()


# Agent 
medicine_price_agent = Agent(name="MedicinePriceAgent", port=5003, endpoint="http://localhost:5003/submit")

# print medicine price agent address
print(medicine_price_agent.address)

# Receiver agent address
# 

@medicine_price_agent.on_message(model=MedicinePriceRequest)
async def handle_medicine_price_request(ctx: Context, sender: str, msg: MedicinePriceRequest):
    """
    Handle the incoming message from the ocr agent.
    Fetch the data for the medicine

    Args:
        ctx (Context): The context object
        sender (str): The sender agent address
        msg (MedicinePriceRequest): The MedicinePriceRequest message
    """

    # Log Context info
    ctx.logger.info(f"Received a request to fetch prices for the medicine: {msg.medicine_name} from {sender}")
    # Initialize the fetcher
    print("⏰️ Initiating Data Fetching...")
    fetcher = FetchMedicinePrices(medicine_name=msg.medicine_name)
    # Search the web for the medicine
    print("🔍️ Searching the web for the medicine...")
    urls = fetcher.fetch_links()
    # Scrape the websites
    print("🕸️ Scraping the websites...")
    pages = asyncio.run(fetcher.fetch_prices(urls))
    # Clean the pages
    print("🧼 Cleaning the pages...")
    cleaned_pages = fetcher.clean_pages(pages)
    # Extract the prices
    print("💰 Extracting prices...")
    start = time.time()
    prices = fetcher.get_prices(cleaned_pages)
    end = time.time()
    print(f"Time taken: {end-start} seconds")

    # Print the prices along with the urls 
    for i, price in enumerate(prices):
        print(f"URL: {urls[i]}")
        print(price)

    # Send the response to the sender
    medicine_price_response = MedicinePriceResponse(
        medicine_name=msg.medicine_name,
        medicine_price_info=prices
    )
    # await ctx.send(sender, medicine_price_response)

