"""
File for agent which takes text reports and images,  finds the anomalies and give the summary.
"""


import pdfplumber
import pytesseract
from PIL import Image
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

# Create the ReportSummarizerAgent (Runs on localhost, port 8002)
report_summarizer_agent = Agent(name="ReportSummarizerAgent", port=8002, endpoint="http://localhost:8002/submit")

# Address of ReportHandlerAgent (Replace with actual ReportHandlerAgent address from logs)
REPORT_HANDLER_AGENT_ADDRESS = "agent1qfteffcpfqhrsj9mpcjxvza42axkr5y9zva0fnmgztzmpaaxse00garhcdv"

'''
Report Processing Handler
'''


@report_summarizer_agent.on_message(model=ReportRequest)
async def process_report(ctx: Context, sender: str, message: ReportRequest):
    """
    Handles incoming report processing requests.
    Extracts text from the given file (PDF/Image) and sends back a response.

    Args:
        ctx (Context): UAgents context for communication
        sender (str): Sender agent's address (ReportHandlerAgent)
        message (ReportRequest): Incoming request containing the file path
    """
    file_path = message.file_path
    ctx.logger.info(f"Received report request from {sender}: {file_path}")

    # Extract text based on file type
    if file_path.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        extracted_text = extract_text_from_image(file_path)
    else:
        extracted_text = "Unsupported file format."

    # Send extracted text back to ReportHandlerAgent
    ctx.logger.info("Sending extracted text back to ReportHandlerAgent...")
    response = ReportResponse(extracted_text=extracted_text)
    await ctx.send(REPORT_HANDLER_AGENT_ADDRESS, response)


'''
Text Extraction Utilities
'''


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Extracted text content or an error message
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

    return text.strip() if text.strip() else "No text found in PDF."


def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        file_path (str): Path to the image file

    Returns:
        str: Extracted text content or an error message
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error extracting text from Image: {str(e)}"

    return text.strip() if text.strip() else "No text found in Image."


'''
Main Execution
'''

if __name__ == "__main__":
    print(f"ReportSummarizerAgent Address: {report_summarizer_agent.address}")  # Print for ReportHandlerAgent reference
    report_summarizer_agent.run()
