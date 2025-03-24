from uagents import Agent, Context 
import asyncio

from workers.fetch_medicine_info import FetchMedicineInfo
from agent_models.medicine_models import MedicineRequest, MedicineResponse


ocr_agent = Agent(name="OCR Agent", port=5002, endpoint="http://localhost:5002/submit")

# Medicine data fetcher agent address 
medicine_agent_address = "agent1qfuf2sf8dczw9pzy059x9wrwq4sdtnf2m5uqgwlttwmgdezvygxpk9sqg6n"


@ocr_agent.on_event("startup")
async def send_request(ctx: Context):
    """
    Send a request to the medicine agent to fetch medicine data.

    Args:             
        ctx (Context): The context object
    """
    # Sample input 
    print("Please enter medicine name: ")
    medicine_name = input()
    
    # Log Context info
    ctx.logger.info("⏰️ Sending a request to the medicine agent to fetch medicine data...")
    
    # Send a request to the medicine agent
    await ctx.send(medicine_agent_address, MedicineRequest(medicine_name=medicine_name))


# run the agent 
if __name__ == "__main__":
    ocr_agent.run()