"""
File for agent which takes the query from the user and forwards it to the required agent.
"""

from uagents import Agent, Context, Model

"""
Request & Response Models
"""

from agent_models.report_models import ReportRequest, ReportResponse
from agent_models.xray_models import XrayRequest, XrayResponse
from agent_models.mri_models import MRIRequest, MRIResponse
from agent_models.lung_models import LungRequest, LungResponse  # <-- Added lung model import

"""
Agent Configuration
"""

report_handler_agent = Agent(name="ReportHandlerAgent", port=8001, endpoint="http://localhost:8001/submit")

# Static agent addresses
REPORT_SUMMARIZER_AGENT_ADDRESS = "agent1qvh0zvv5snymgtx43af5pcnkp2sunm0pfr68j53flmquuu7jd80kgdrr0h7"
CHEST_XRAY_AGENT_ADDRESS = "agent1qdpsw05ma4wd92647zssdyd2hvyu2qfffn39y7yqvstpf8vw6syz2l9c869"
MRI_AGENT_ADDRESS = "agent1qf0mqfr25jtrarxs3jh9t4xl42snz5k6q7nfnjj4v8qgy8phzd75y689zry"
LUNG_AGENT_ADDRESS = "agent1qw3adtswm99rnmah2gapuq0pelaqkhhe7f5qte2c088m062uuupqcuq5fuy"  # <-- Added lung agent address

"""
Startup Handler with User Options
"""

@report_handler_agent.on_event("startup")
async def send_request(ctx: Context):
    print("\nSelect the type of file to process:")
    print("1. Text Report (PDF/Image)")
    print("2. Medical Image (X-ray, MRI, CT scan, etc.)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        file_path = input("Enter the path to the medical report (PDF/Image): ").strip()
        request = ReportRequest(file_path=file_path)
        ctx.logger.info(f"Sending report to ReportSummarizerAgent with path: {file_path}")
        await ctx.send(REPORT_SUMMARIZER_AGENT_ADDRESS, request)

    elif choice == "2":
        print("\nSelect type of medical image:")
        print("1. Chest X-ray")
        print("2. Brain MRI")
        print("3. Lung CT Scan")  # <-- Added lung option
        image_choice = input("Enter 1, 2, or 3: ").strip()
        file_path = input("Enter the path to the medical image file: ").strip()

        if image_choice == "1":
            request = XrayRequest(file_path=file_path)
            ctx.logger.info(f"Sending X-ray to ChestXrayAgent with path: {file_path}")
            await ctx.send(CHEST_XRAY_AGENT_ADDRESS, request)

        elif image_choice == "2":
            request = MRIRequest(file_path=file_path)
            ctx.logger.info(f"Sending MRI to BrainMRIAgent with path: {file_path}")
            await ctx.send(MRI_AGENT_ADDRESS, request)

        elif image_choice == "3":
            request = LungRequest(file_path=file_path)
            ctx.logger.info(f"Sending Lung CT to LungCancerAgent with path: {file_path}")
            await ctx.send(LUNG_AGENT_ADDRESS, request)

        else:
            ctx.logger.error("Invalid image type selection.")

    else:
        ctx.logger.error("Invalid option. Please enter 1 or 2.")

"""
Response Handlers
"""

@report_handler_agent.on_message(model=ReportResponse)
async def handle_report_response(ctx: Context, sender: str, message: ReportResponse):
    ctx.logger.info(f"\nReceived Summary from {sender}:\n{message.extracted_text}")

@report_handler_agent.on_message(model=XrayResponse)
async def handle_xray_response(ctx: Context, sender: str, message: XrayResponse):
    ctx.logger.info(f"\nReceived X-ray Analysis from {sender}:\n{message.detected_conditions}")

@report_handler_agent.on_message(model=MRIResponse)
async def handle_mri_response(ctx: Context, sender: str, message: MRIResponse):
    ctx.logger.info(f"\nReceived Brain MRI Prediction from {sender}: {message.tumor_prediction}")

@report_handler_agent.on_message(model=LungResponse)  # <-- Added Lung response handler
async def handle_lung_response(ctx: Context, sender: str, message: LungResponse):
    ctx.logger.info(f"\nReceived Lung CT Prediction from {sender}: {message.cancer_prediction}")

"""
Main Execution
"""

if __name__ == "__main__":
    print(f"ReportHandlerAgent Address: {report_handler_agent.address}")
    report_handler_agent.run()
