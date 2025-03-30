from uagents import Model

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
