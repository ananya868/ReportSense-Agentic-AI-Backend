from fetch_medicine_data import FetchMedicineData
from agent_models.medicine_models import MedicineRequest, MedicineResponse

from uagents import Agent, Context 
import asyncio

medicine_agent = Agent(name="MedicineAgent", port=5000, endpoint="http://localhost:5000/submit")

# Receiver agent address | LLM Medicine Informant Agent  
llm_medicine_informant_address = "##"

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
    ctx.logger.info(f"Received a request to fetch data for the medicine: {msg.medicine_name}")

    # Initialize the fetcher
    fetcher = FetchMedicineData(medicine_name=msg.medicine_name)
    # Search the web for the medicine
    url = fetcher.search_web()
    # Fetch the webpage content
    page = asyncio.run(fetcher.fetch_webpage(url))
    context = page[0]

    # Build prompts with context
    prompt = prompt.format(context=context)
    sys_prompt = sys_prompt
    
    # Generate medicine data points 
    data_points = fetcher.generate_data_points(prompt, sys_prompt)
    # Convert to json/dict
    data_dict = data_points.dict()
    if msg.is_save: # Save the data to a json file
        with open(f"medi_data/{msg.medicine_name}_data.json", "w") as f: 
            json.dump(data_dict, f, indent=4)
    
    # Send the data to the LLM Medicine Informant Agent
    medicine_response = MedicineResponse(
        medicine_name=msg.medicine_name,
        medicine_data=data_dict
    )
    await ctx.send(llm_medicine_informant_address, medicine_response)


