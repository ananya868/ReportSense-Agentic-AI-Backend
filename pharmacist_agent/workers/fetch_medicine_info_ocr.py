import base64 
from openai import OpenAI 
import json 
import os 


class FetchMedicineInfo:
    """
    A class for fetching medicine information using OCR.
    """

    # The prompt for the GPT-4o model
    prompt = """ 
        The following is an image of a medical prescription. 
        Analyze the image carefully and extract names of the medicines. 
        Only return the names of the medicines in the prescription in the format:
        {"medicines": ["medicine_1", "medicine_2", ...]} and nothing extra!.
        If you are unable to extract any medicines, return an empty list.
    """ 

    def __init__(self, api_key: str = os.getenv("OPENAI_API_KEY")): 
        self.api_key = api_key
    
    def encode_image(self, image_path: str): 
        """
        Encode an image file to base64.

        Args:
            image_path (str): The path to the image file
        
        Returns:
            str: The base64 encoded image
        """
        with open(image_path, "rb") as image_file: 
            return base64.b64encode(image_file.read()).decode("utf-8")

    def fetch_medicine_info(self, image_path: str) -> dict:
        """
        Fetch medicine information using OCR.

        Args:
            image_path (str): The path to the image file
        
        Returns:
            dict: The extracted medicine information
        """
        # Initialize the OpenAI client
        client = OpenAI(api_key=self.api_key)

        # Encode the image 
        encoded_image = self.encode_image(image_path)

        # Generate the prompt 
        prompt_msg = self.prompt 

        # Generate the completion
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt_msg },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
        )

        response = completion.choices[0].message.content 
        return json.loads(response)
        



