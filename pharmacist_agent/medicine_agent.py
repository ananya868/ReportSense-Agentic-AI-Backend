from workers.fetch_medicine_data import FetchMedicineData
from agent_models.medicine_models import MedicineRequest, MedicineResponse

from uagents import Agent, Context 
import asyncio
import json 
import os 

import nest_asyncio
nest_asyncio.apply()


# Agent 
medicine_agent = Agent(name="MedicineAgent", port=5000, endpoint="http://localhost:5000/submit")

# Medicine data fetcher agent address | Sender Agent | This agent
# print(medicine_agent.address)

# Receiver agent address | LLM Medicine Informant Agent | Receiver Agent 
llm_medicine_informant_address = "agent1qfw83q7u9wzc04c5t6f5a4evhkexwkfxfatrvln6wk0hmn9myrkmsz9uqfy"

# Define the prompt and system prompt 
prompt = """Analyze the provided webpage content and extract structured details about the medication using the following fields:
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
    Ensure accuracy while extracting details and avoid making assumptions. Only use information explicitly stated in the context.
"""


@medicine_agent.on_message(model=MedicineRequest)
async def handle_medicine_request(ctx: Context, sender: str, msg: MedicineRequest):
    """
    Handle the incoming message from the ocr agent.
    Fetch the data for the medicine
    Sends the data to the LLM Medicine Informant Agent

    Args: 
        ctx (Context): The context object
        sender (str): The sender agent address
        msg (MedicineRequest): The MedicineRequest message
    """
    
    # Log Context info
    ctx.logger.info(f"Received a request to fetch data for the medicine: {msg.medicine_name} from {sender}")

    # Initialize the fetcher 
    if msg.verbose:
        print("Initiating Data Fetching...")
    fetcher = FetchMedicineData(medicine_name=msg.medicine_name)
    
    # Search the web for the medicine
    if msg.verbose:
        print("Searching the web for the medicine...")
    url = fetcher.search_web()
    
    # Fetch the webpage content
    if msg.verbose:
        print("Fetching the webpage content...")
    page = asyncio.run(fetcher.fetch_webpage(url))
    context = page[0]
    if msg.verbose:
        print("Webpage content fetched successfully!")

    # Build prompts with context
    formatted_prompt = prompt.format(context=context)
    formatted_sys_prompt = sys_prompt
    
    # Generate medicine data points 
    if msg.verbose:
        print("Generating Data Points...")
    data_points = fetcher.generate_data_points(formatted_prompt, formatted_sys_prompt)
    # Convert to json/dict
    data_dict = data_points.dict()

    if msg.verbose:
        print(f"Saving status: {msg.is_save}")
    
    if msg.is_save: # Save the data to a json file
        with open(f"medi_data/{msg.medicine_name}_data.json", "w") as f: 
            json.dump(data_dict, f, indent=4)
    
    if msg.verbose:
        print("Data Points Generated Successfully")

    # Send the data to the LLM Medicine Informant Agent
    medicine_response = MedicineResponse(
        medicine_name=msg.medicine_name,
        medicine_info=data_dict
    )
    await ctx.send(llm_medicine_informant_address, medicine_response)


# Run the agent
if __name__ == "__main__":
    medicine_agent.run()
