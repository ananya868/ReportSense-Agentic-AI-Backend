"""
Medicine OCR Agent
===================

Medicine OCR Agent receives a request from Test Agent (Manager Agent) to fetch medicine names from an image.
Uses vision model to detect the medicine names from the image.
The detected names are then confirmed by the user.
Users can also manually enter the medicine names if OCR fails to detect them.
This agent sends the detected/entered medicine names to the Medicine Finder Agent or Medicine Data Agent. 
"""

from uagents import Agent, Context 
import asyncio
import os
from dotenv import load_dotenv
from typing import Any 

from workers.medicine_ocr_worker import MedicineOCR
from agent_models.medicine_models import MedicineOCRRequest, MedicineOCRResponse

load_dotenv()


"""Agent"""
ocr_agent = Agent(name="OCR Agent", port=5002, endpoint="http://localhost:5002/submit")

"""Agent Address"""
ocr_agent_address = "agent1q0xe4wfkjuk3zc6v5683cyfa7gk38erd2lzt49pejmryv822x084wqeygc2"

"""Medicine data fetcher agent address or Medicine finder agent"""
medicine_data_agent_address = "agent1qfuf2sf8dczw9pzy059x9wrwq4sdtnf2m5uqgwlttwmgdezvygxpk9sqg6n"
medicine_finder_agent = "agent1qgfx3g350nc4gqrguhfqr0hxv9zx72urq6jhfatf3s765rhzncjc2wcssnq"

"""Worker"""  
med_ocr = MedicineOCR(api_key = os.getenv("OPENAI_API_KEY"))


"""Function Definitions"""
def get_manual_entry() -> list:
    """
    Get medicine names through manual user input.
    
    Returns:
        list: List of medicine names entered by user
    """
    while True:
        entry_type = input("Do you want to enter one medicine name or multiple? (one/multiple): ").strip().lower()
        if entry_type not in ["one", "multiple"]:
            print("Invalid choice. Please enter 'one' or 'multiple'.")
            continue
        
        user_input = input("Enter medicine name(s), use commas if entering multiple: ")
        if entry_type == "multiple" and "," not in user_input:
            print("You selected multiple but did not use commas. Please separate names correctly.")
            continue
        
        medicines = [med.strip() for med in user_input.split(",") if med.strip()]
        return medicines

def confirm_medicines(medicines) -> Any:
    """
    Ask user to confirm the detected/entered medicine names.
    
    Args:
        medicines (list): List of medicine names to confirm
        
    Returns:
        tuple: (bool, list) - (Whether medicines are confirmed, confirmed medicine list)
    """
    if not medicines:
        return False, []
        
    print(f"Please confirm the following medicines: {', '.join(medicines)}")
    confirmation = input("Are these correct? (yes/no): ").strip().lower()
    
    return confirmation == "yes", medicines

def process_medicines(img_path) -> list:
    """
    Main function to handle medicine detection workflow.

    Args:
        img_path (str): Path to the image containing medicine names.

    Returns:
        list: Final list of confirmed medicines
    """
    medicines = []
    confirmed = False
    
    # Verify if the image path is valid
    if not img_path or not isinstance(img_path, str):
        print("Invalid image path. Please provide a valid path.")
        return medicines
    if not img_path.endswith(('.png', '.jpg', '.jpeg')):
        print("Invalid image format. Please provide a .png, .jpg, or .jpeg file.")
        return medicines
    
    while not confirmed:
        method = input("Please select method to fetch medicine names: \n 1. OCR \n 2. Manual Entry \n => ")
        
        if method == "1":
            medicines = med_ocr.fetch(img_path)
            if not medicines:
                print(
                    "No medicines found! Please select one out of the following: \n",
                    "1. Enter medicine(s) name manually \n",
                    "2. Retry the OCR process \n",
                )
                choice = input("Please select an option (1 or 2) => ").strip()
                if choice == "1":
                    medicines = get_manual_entry()
                elif choice == "2":
                    medicines = med_ocr.fetch(img_path)
        
        elif method == "2":
            medicines = get_manual_entry()
        
        else:
            print("Invalid option. Please select 1 or 2.")
            continue
            
        if not medicines:
            print("No valid medicines found!")
            retry = input("Would you like to try again? (yes/no) => ").strip().lower()
            if retry != "yes":
                return []
            continue
            
        confirmed, medicines = confirm_medicines(medicines)
        
        if not confirmed:
            print("Medicine names not confirmed. Let's try again.")
    
    # print(f"Final medicines confirmed: {medicines}")
    return medicines


"""Define the agent"""
# @ocr_agent.on_message(Model=MedicineOCRRequest) # to be implemented ... 
@ocr_agent.on_event("startup")
async def send_request(ctx: Context):
    """
    Send a request to the medicine agent to fetch medicine data.

    Args:             
        ctx (Context): The context object
    """
    img_path = "med.png" ## This will be fetched from msg.img_path | MedicineOCRRequest
    medicines = process_medicines(img_path)
    if medicines: 
        ctx.logger.info(f"Medicines detected: {medicines}")
    else: 
        ctx.logger.info("Process terminate without valid medicines!")
    # Log Context info
    ctx.logger.info("⏰️ Sending a request to the medicine agent to fetch medicine data...")

    # Send a request to the medicine agent
    # await ctx.send(medicine_agent_address, MedicineRequest(medicine_name=medicine_name))



# run the agent 
if __name__ == "__main__":
    ocr_agent.run()
    # pass