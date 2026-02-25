import os
from google import genai
from PIL import Image

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "dummy"))
# Just testing the types
print(type(client.models.generate_content))
print("Imported genai successfully")
