"""
File for agent which takes text chest xray images and predict the diseases
"""


import torch
import torchvision.transforms as transforms
from PIL import Image
from uagents import Agent, Context
from torchvision import models

'''
Request & Response Models
- XrayRequest: contains the file path of the chest X-ray image sent to the agent
- XrayResponse: returns the detected conditions with confidence scores
'''
import sys 
# Add the parent directory to the path
sys.path.append("..")
from agent_models.xray_models import XrayRequest, XrayResponse


'''
Agent Configuration
'''

# Create the ChestXrayAgent (Runs locally on port 8003)
chest_xray_agent = Agent(name="ChestXrayAgent", port=8003, endpoint="http://localhost:8003/submit")

# Address of the ReportHandlerAgent that will receive the analysis results
REPORT_HANDLER_AGENT_ADDRESS = "agent1qfteffcpfqhrsj9mpcjxvza42axkr5y9zva0fnmgztzmpaaxse00garhcdv"

'''
Load Pre-trained CheXNet Model (DenseNet-121)
'''

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DenseNet-121 model architecture without pretrained classifier
model = models.densenet121(pretrained=False)

# Class labels for ChestX-ray14 dataset (14 disease conditions)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax"
]

# Modify final classifier layer to output 14 disease probabilities
model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))

# Load model weights (ignoring mismatched layers like old classifier)
state_dict = torch.load("C:/Users/91790/Desktop/Projects/ReportSense-Agentic-AI-Backend/diagnosis-agent/image_models/weights/chexnet_model.pth", map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}  # Remove outdated classifier weights
model.load_state_dict(filtered_state_dict, strict=False)

# Move model to appropriate device and set to evaluation mode
model = model.to(device)
model.eval()

'''
Chest X-ray Processing Handler
- This function is triggered when a message of type XrayRequest is received.
- It runs the image through CheXNet and returns predictions.
'''

@chest_xray_agent.on_message(model=XrayRequest)
async def analyze_xray(ctx: Context, sender: str, message: XrayRequest):
    """
    Handles incoming chest X-ray analysis requests.
    Processes the image using CheXNet CNN and returns the detected conditions.

    Args:
        ctx (Context): UAgents context for communication
        sender (str): Sender agent's address (ReportHandlerAgent)
        message (XrayRequest): Incoming request containing the image file path
    """
    file_path = message.file_path
    ctx.logger.info(f"Received chest X-ray analysis request from {sender}: {file_path}")

    # Analyze the X-ray and return disease probabilities
    detected_conditions = classify_xray(file_path)

    # Send the analysis results back to the ReportHandlerAgent
    ctx.logger.info(f"Sending analysis result to ReportHandlerAgent: {detected_conditions}")
    response = XrayResponse(detected_conditions=detected_conditions)
    await ctx.send(REPORT_HANDLER_AGENT_ADDRESS, response)

'''
Chest X-ray Multi-Label Classification Function
- This function takes an image path, processes it, runs inference,
  and returns disease predictions with confidence scores.
'''

def classify_xray(file_path: str):
    """
    Classifies a chest X-ray image using the CheXNet model.
    Returns multiple detected conditions with confidence scores.

    Args:
        file_path (str): Path to the X-ray image

    Returns:
        dict: Dictionary of detected conditions with confidence scores
    """
    try:
        # Load and preprocess the image
        image = Image.open(file_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match CheXNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)

        # Run inference without computing gradients
        with torch.no_grad():
            output = model(image)
            probabilities = torch.sigmoid(output[0])  # Sigmoid for multi-label classification

        # Only return conditions with probability > 50%
        detected_conditions = {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES)) if probabilities[i].item() > 0.5
        }

        # Return results or default message
        return detected_conditions if detected_conditions else {"No disease detected": 0.0}

    except Exception as e:
        return {"Error processing image": str(e)}

'''
Main Execution
- Prints the agent's address and starts the agent server.
'''

if __name__ == "__main__":
    print(f"ChestXrayAgent Address: {chest_xray_agent.address}")
    chest_xray_agent.run()
