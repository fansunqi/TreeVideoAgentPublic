# 25/02/13 fixed
from together import Together
import base64

client = Together(api_key="639ffdc334c02d1d9032c3d9738e685c6073536ebd70273ec9dec6509db44577")

getDescriptionPrompt = "What is in the image?"

imagePath= "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/output_frames/250_0b4529ac-5a4e-4d30-b6b6-c6504c509c0c/frame_0.jpg"

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(imagePath)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": getDescriptionPrompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)