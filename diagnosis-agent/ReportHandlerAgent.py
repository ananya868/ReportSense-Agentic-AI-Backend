"""
File for agent which takes the query from the user and forwards it to the required agent.
"""

from uagents import Agent, Context, Model

'''
Request & Response Models
- ReportRequest: contains the file path of the medical report (PDF or image)
- ReportResponse: returns the extracted and summarized text from the report
- XrayRequest: contains the file path of the chest X-ray image sent for analysis
- XrayResponse: returns detected conditions from the X-ray with confidence scores
'''
from agent_models.report_models import ReportRequest, ReportResponse
from agent_models.xray_models import XrayRequest, XrayResponse


'''
Agent Configuration
- ReportHandlerAgent runs locally on port 8001 and sends requests
  to either ReportSummarizerAgent or ChestXrayAgent based on user input.
'''

report_handler_agent = Agent(name="ReportHandlerAgent", port=8001, endpoint="http://localhost:8001/submit")

# Static agent addresses (replace MRI one when available)
REPORT_SUMMARIZER_AGENT_ADDRESS = "agent1qvh0zvv5snymgtx43af5pcnkp2sunm0pfr68j53flmquuu7jd80kgdrr0h7"
CHEST_XRAY_AGENT_ADDRESS = "agent1qdpsw05ma4wd92647zssdyd2hvyu2qfffn39y7yqvstpf8vw6syz2l9c869"
MRI_AGENT_ADDRESS = "agent1qmri..."  # Placeholder address for MRI analysis agent (to be implemented)


'''
Startup Handler with User Options
- This runs once on agent startup and asks user to select the type of report.
- Based on input, it sends the appropriate request to the corresponding agent.
'''

@report_handler_agent.on_event("startup")
async def send_request(ctx: Context):
    print("\nSelect the type of file to process:")
    print("1. Text Report (PDF/Image)")
    print("2. Medical Image (X-ray, MRI, etc.)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Handle medical text report
        file_path = input("Enter the path to the medical report (PDF/Image): ").strip()
        request = ReportRequest(file_path=file_path)
        ctx.logger.info(f"Sending report to ReportSummarizerAgent with path: {file_path}")
        await ctx.send(REPORT_SUMMARIZER_AGENT_ADDRESS, request)

    elif choice == "2":
        # Handle medical image
        print("\nSelect type of medical image:")
        print("1. Chest X-ray")
        print("2. MRI (to be implemented)")
        image_choice = input("Enter 1 or 2: ").strip()
        file_path = input("Enter the path to the medical image file: ").strip()
        request = XrayRequest(file_path=file_path)

        if image_choice == "1":
            ctx.logger.info(f"Sending X-ray to ChestXrayAgent with path: {file_path}")
            await ctx.send(CHEST_XRAY_AGENT_ADDRESS, request)
        elif image_choice == "2":
            ctx.logger.info(f"Sending MRI to MRIAgent with path: {file_path}")
            await ctx.send(MRI_AGENT_ADDRESS, request)
        else:
            ctx.logger.error("Invalid image type selection.")
    else:
        ctx.logger.error("Invalid option. Please enter 1 or 2.")


'''
Response Handlers
- Handles the response from either ReportSummarizerAgent or ChestXrayAgent
'''

@report_handler_agent.on_message(model=ReportResponse)
async def handle_report_response(ctx: Context, sender: str, message: ReportResponse):
    ctx.logger.info(f"\nReceived Summary from {sender}:\n{message.extracted_text}")


@report_handler_agent.on_message(model=XrayResponse)
async def handle_xray_response(ctx: Context, sender: str, message: XrayResponse):
    ctx.logger.info(f"\nReceived X-ray/MRI Analysis from {sender}:\n{message.detected_conditions}")


'''
Main Execution
- Starts the ReportHandlerAgent and prints its address.
'''

if __name__ == "__main__":
    print(f"ReportHandlerAgent Address: {report_handler_agent.address}")
    report_handler_agent.run()
