"""
Medicine Price Agent
===================

This agent is responsible for fetching the prices of medicines from various pharmacy websites.
"""

from workers.medicine_finder_agent import FetchMedicinePrices
from agent_models.medicine_price_models import MedicinePriceRequest, MedicinePriceResponse, MedicinePriceInfo

from uagents import Agent, Context
import asyncio
import json
import os, time

import nest_asyncio
nest_asyncio.apply()


"""Agent""" 
medicine_price_agent = Agent(name="MedicinePriceAgent", port=5003, endpoint="http://localhost:5003/submit")

"""Agent Address"""
medicine_finder_agent_address = "agent1qgfx3g350nc4gqrguhfqr0hxv9zx72urq6jhfatf3s765rhzncjc2wcssnq"

"""Receiver agent address | Chatbot agent"""
# chatbot_agent_address = "nothing"


"""Define the agent"""
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
    prices_info = {}
    for i in msg.medicine_names: 
        # Log Context info
        ctx.logger.info(f"Received a request to fetch prices for the medicine: {i} from {sender}")
        # Initialize the fetcher
        fetcher = FetchMedicinePrices(medicine_name=i)
        # Search the web for the medicine
        urls = fetcher.fetch_links()
        # Scrape the websites
        pages = asyncio.run(fetcher.fetch_prices(urls))
        # Clean the pages
        cleaned_pages = fetcher.clean_pages(pages)
        # Extract the prices
        prices = fetcher.get_prices(cleaned_pages, provider = "openai") 
        # Store the prices in the dictionary
        prices_info[i] = prices

    # Create the response object 
    medicine_price_response = MedicinePriceResponse(medicine_price_info=prices_info)

    # print("Print Schema")
    # print(medicine_price_response.json())
    # await ctx.send(sender, medicine_price_response)


if __name__ == "__main__":
    # medicine_price_agent.run()
    pass 