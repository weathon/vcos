from openai import OpenAI
import dotenv 
import os
import json
from PIL import Image
import numpy as np  
dotenv.load_dotenv()

from pydantic import BaseModel, Field
class Response(BaseModel): 
    bbox: list = Field(..., description="The bounding box.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

import signal
import time

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def with_timeout_and_retry(func, retries=10, timeout=15):
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except TimeoutException:
                signal.alarm(0)
                print("timeout trying")
                if attempt == retries - 1:
                    raise TimeoutException("Function timed out after all retries")
                time.sleep(1)
    return wrapper

import base64
from io import BytesIO

def encode_image_for_llm(image):
    if isinstance(image, Image.Image): 
        pil_image = image
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            pil_image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path"):
            pil_image = Image.open(image["path"])
        else:
            raise ValueError("image dictionary missing both 'bytes' and 'path'")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def det_(pil_blended, flow):
    while 1:
        try:
            completion = client.chat.completions.parse(
            model="qwen/qwen3-vl-235b-a22b-instruct",
            messages=[
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "text",
                                "text": "You will get two images, one is a highlighted normal image, and another is an optical flow image. Please find a camouflaged animal/insect in this image and provide the bounding box. It MIGHT be highlighted in blue in the original image and show up on the flow map. If you cannot see it, return an empty list; do NOT return anything else as 'not found'. But the thing is definitely there, so you should be able to find it. If you cannot find it in original image, use the flow map to find the place moved the most. Note that not all moving parts are the target, only the animal itself and not shadows or reflections."
                            },
                            {
                                "type": "image_url", 
                                "image_url": { 
                                "url": encode_image_for_llm(pil_blended)
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": { 
                                "url": encode_image_for_llm(flow)
                                }
                            }
                            ]
                        }
                        ],
            response_format=Response,
            temperature=0.0, 
            )
            # print(completion.choices[0].message.content) 
            ans = json.loads(completion.choices[0].message.content)
            if ans is None:
                return None
            bounding_box = ans["bbox"]
            if len(bounding_box) != 4 or bounding_box is None or sum(bounding_box) == 0:
                return None
            height, width, _ = np.array(pil_blended).shape
            abs_y1 = int(bounding_box[1] / 1000 * height) 
            abs_x1 = int(bounding_box[0] / 1000 * width)
            abs_y2 = int(bounding_box[3] / 1000 * height)
            abs_x2 = int(bounding_box[2] / 1000 * width)
            return abs_x1, abs_y1, abs_x2, abs_y2
        except Exception as e:
            # pass
            print("Error occurred:", e)
            print("Retrying...")

@with_timeout_and_retry
def det(pil_blended, flow):
    return det_(pil_blended, flow)