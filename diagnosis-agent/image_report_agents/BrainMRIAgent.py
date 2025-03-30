"""
File for agent which takes brain MRI images and predicts tumor type
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from uagents import Agent, Context

'''
Request & Response Models
- MRIRequest: contains the file path of the brain MRI image sent to the agent
- MRIResponse: returns the predicted tumor type as a string
'''
from agent_models.mri_models import MRIRequest, MRIResponse

'''
Agent Configuration
'''

# Create the BrainMRIAgent (Runs locally on port 8004)
brain_mri_agent = Agent(name="BrainMRIAgent", port=8004, endpoint="http://localhost:8004/submit")

# Address of the ReportHandlerAgent that will receive the analysis results
REPORT_HANDLER_AGENT_ADDRESS = "agent1qfteffcpfqhrsj9mpcjxvza42axkr5y9zva0fnmgztzmpaaxse00garhcdv"

'''
Load Pre-trained Brain Tumor Classification Model (EfficientNet-B3)
'''

# Path to trained model weights
MODEL_PATH = "/Users/js/Desktop/ReportSense-Agentic-AI-Backend/diagnosis-agent/image_models/weights/brain_mri_model.pt"

# Class labels for brain tumor classification
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Use GPU (Apple MPS) if available, otherwise fallback to CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load EfficientNet-B3 model and adjust output layer
model = EfficientNet.from_name('efficientnet-b3')
model._fc = nn.Linear(model._fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

'''
Image Preprocessing Pipeline
'''

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize to match EfficientNet-B3 input
    transforms.Grayscale(num_output_channels=3),  # Ensure 3-channel input
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # Normalize image
])

'''
Brain MRI Analysis Handler
- Triggered when MRIRequest is received.
- Runs the image through the model and returns predicted tumor type.
'''

@brain_mri_agent.on_message(model=MRIRequest)
async def analyze_mri(ctx: Context, sender: str, message: MRIRequest):
    """
    Handles incoming brain MRI analysis requests.
    Processes the image using the brain tumor CNN model and returns the prediction.

    Args:
        ctx (Context): UAgents context for communication
        sender (str): Sender agent's address (ReportHandlerAgent)
        message (MRIRequest): Incoming request containing the image file path
    """
    file_path = message.file_path
    ctx.logger.info(f"Received brain MRI analysis request from {sender}: {file_path}")

    prediction = classify_mri(file_path)

    ctx.logger.info(f"Sending analysis result to ReportHandlerAgent: {prediction}")
    response = MRIResponse(tumor_prediction=prediction)
    await ctx.send(REPORT_HANDLER_AGENT_ADDRESS, response)

'''
Brain MRI Tumor Classification Function
- Takes a file path, transforms the image, runs inference,
  and returns the predicted tumor class.
'''

def classify_mri(file_path: str):
    """
    Classifies a brain MRI image using the EfficientNet-B3 model.

    Args:
        file_path (str): Path to the brain MRI image

    Returns:
        str: Predicted tumor type or error message
    """
    try:
        image = Image.open(file_path).convert("L")  # Ensure grayscale input
        image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            predicted_class = CLASS_NAMES[pred.item()]

        return predicted_class

    except Exception as e:
        return f"Error processing image: {str(e)}"

'''
Main Execution
- Prints the agent's address and starts the agent server.
'''

if __name__ == "__main__":
    print(f"BrainMRIAgent Address: {brain_mri_agent.address}")
    brain_mri_agent.run()