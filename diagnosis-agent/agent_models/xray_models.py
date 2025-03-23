from uagents import Model

class XrayRequest(Model):
    """
    Request model containing the file path of the chest X-ray image.
    Sent from ReportHandlerAgent to ChestXrayAgent.
    """
    file_path: str  # Path to the chest X-ray image

class XrayResponse(Model):
    """
    Response model containing the detected conditions and confidence scores.
    Sent from ChestXrayAgent back to ReportHandlerAgent.
    """
    detected_conditions: dict  # Dictionary of detected diseases & confidence scores
