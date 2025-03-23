from uagents import Agent, Context 
import asyncio

from agent_models.medicine_models import MedicineRequest, MedicineResponse


llm_medicine_informant_agent = Agent(name="LLMMedicineInformantAgent", port=5001, endpoint="http://localhost:5001/submit")

# print llm_medicine_informant_agent address
print(llm_medicine_informant_agent.address)

# Receiver agent address | Medicine Agent
medicine_agent_address = "##"


@llm_medicine_informant_agent.on_message(model=MedicineResponse)
async def handle_medicine_response(ctx: Context, sender: str, msg: MedicineResponse):
    """
    Handle the incoming message from the LLM Medicine Informant Agent.

    Args:
        ctx (Context): The context object
        sender (str): The sender agent address
        msg (MedicineResponse): The MedicineResponse message
    """

    # Log Context info
    ctx.logger.info(f"Received a response from the LLM Medicine Informant Agent for the medicine: {msg.medicine_name} from {sender}")

    # print the information extracted from the context
    print(msg.medicine_data)


# Run the agent
if __name__ == "__main__":
    llm_medicine_informant_agent.run()