"""
File for agent which takes the query from the user and forwards it to the required agent.

"""

from uagents import Agent, Context, Model

'''
Request & Response Models
'''


class ReportRequest(Model):
    """
    Request model containing the file path of the medical report.
    Sent from ReportHandlerAgent to ReportSummarizerAgent.
    """
    file_path: str  # Path to the medical report (PDF/Image)


class ReportResponse(Model):
    """
    Response model containing the extracted text from the medical report.
    Sent from ReportSummarizerAgent back to ReportHandlerAgent.
    """
    extracted_text: str  # Extracted text content


'''
Agent Configuration
'''

# Create the ReportHandlerAgent (Runs on localhost, port 8001)
report_handler_agent = Agent(name="ReportHandlerAgent", port=8001, endpoint="http://localhost:8001/submit")

# Address of ReportSummarizerAgent (Replace with actual address from logs)
REPORT_SUMMARIZER_AGENT_ADDRESS = "agent1qvh0zvv5snymgtx43af5pcnkp2sunm0pfr68j53flmquuu7jd80kgdrr0h7"

'''
Send Report Request Once on Startup
'''


@report_handler_agent.on_event("startup")
async def send_request_once(ctx: Context):
    """
    Sends a report file path to ReportSummarizerAgent once on agent startup.
    """
    request = ReportRequest(file_path="/Users/js/Documents/report.pdf")
    ctx.logger.info(f"Sending report request to {REPORT_SUMMARIZER_AGENT_ADDRESS}...")
    await ctx.send(REPORT_SUMMARIZER_AGENT_ADDRESS, request)


'''
Response Handling
'''


@report_handler_agent.on_message(model=ReportResponse)
async def receive_response(ctx: Context, sender: str, message: ReportResponse):
    """
    Handles incoming extracted text responses from ReportSummarizerAgent.
    """
    ctx.logger.info(f"Received Extracted Text from {sender}: {message.extracted_text}")


'''
Main Execution
'''

if __name__ == "__main__":
    print(f"ReportHandlerAgent Address: {report_handler_agent.address}")  # Print for ReportSummarizerAgent reference
    report_handler_agent.run()
