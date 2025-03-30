"""
File for agent which takes text reports and images, finds the anomalies and gives the summary.
"""

import pdfplumber
import pytesseract
from PIL import Image

from uagents import Agent, Context

from openai import OpenAI
from dotenv import load_dotenv

'''
Request & Response Models
- ReportRequest: contains the file path of the medical report (PDF or image)
- ReportResponse: returns the extracted and summarized text from the report
'''
from agent_models.report_models import ReportRequest, ReportResponse


'''
Load API Key
- Load OpenAI API key from the .env file into environment variables.
'''

load_dotenv()

'''
Agent Configuration
- ReportSummarizerAgent runs locally on port 8002.
- It receives text/image reports, extracts content, sends it to GPT-3.5, and returns a medical summary.
'''

report_summarizer_agent = Agent(name="ReportSummarizerAgent", port=8002, endpoint="http://localhost:8002/submit")

# Address of the ReportHandlerAgent that will receive the summarized report
REPORT_HANDLER_AGENT_ADDRESS = "agent1qfteffcpfqhrsj9mpcjxvza42axkr5y9zva0fnmgztzmpaaxse00garhcdv"

'''
Report Processing Handler
- Triggered when a ReportRequest is received.
- Extracts text from the report file (PDF or image), generates summary using GPT-3.5, and responds.
'''

@report_summarizer_agent.on_message(model=ReportRequest)
async def process_report(ctx: Context, sender: str, message: ReportRequest):
    """
    Handles incoming report processing requests.
    Extracts text from the given file (PDF/Image), sends it to GPT-3.5,
    and sends back the summarized result with abnormalities.

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

    # Send to GPT for summarization & abnormality detection
    ctx.logger.info("Sending extracted text to GPT-3.5 for medical summary...")
    summarized_text = summarize_with_gpt(extracted_text)

    # Send summarized output back to ReportHandlerAgent
    ctx.logger.info("Sending summarized result back to ReportHandlerAgent...")
    response = ReportResponse(extracted_text=summarized_text)
    await ctx.send(REPORT_HANDLER_AGENT_ADDRESS, response)


'''
Text Extraction Utilities
- Extracts text from either PDF or image using pdfplumber or pytesseract respectively.
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
OpenAI GPT-3.5 Integration
- Uses OpenAI’s chat completion API to analyze and summarize the medical report.
'''

client = OpenAI()  # Automatically reads API key from environment

def summarize_with_gpt(report_text: str) -> str:
    """
    Sends the extracted report text to OpenAI GPT-3.5 for summarization and abnormality detection.

    Args:
        report_text (str): The extracted raw text from the report

    Returns:
        str: A natural language summary with abnormalities and simplified explanations
    """
    try:
        prompt = f"""
You are a medical AI assistant. Analyze the following medical report.
1. Summarize the findings.
2. Highlight any abnormal values or potential health concerns.
3. Explain complex medical terms in simple language.

Report:
\"\"\"
{report_text}
\"\"\"
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary with GPT: {str(e)}"


'''
Main Execution
- Starts the ReportSummarizerAgent and prints its address for reference.
'''

if __name__ == "__main__":
    print(f"ReportSummarizerAgent Address: {report_summarizer_agent.address}")
    report_summarizer_agent.run()
