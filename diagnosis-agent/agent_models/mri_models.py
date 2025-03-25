from uagents import Model

# Request sent to BrainMRIAgent
class MRIRequest(Model):
    file_path: str  # Full path to the MRI image to classify

# Response returned by BrainMRIAgent
class MRIResponse(Model):
    tumor_prediction: str  # Predicted tumor type label

