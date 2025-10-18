import asyncio
import base64
import io

import aiohttp
from PIL import Image

IMAGE_TOKEN_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def create_image_bytes(color, width=512, height=512, format="PNG"):
    img = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def one_image_request(image_data, request_num):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Hello {IMAGE_TOKEN_TEXT}",
                "image_data": image_data,
                "sampling_params": {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "stop_token_ids": [1],
                },
            },
        )
        response_json = await response.json()
        print(f"Request {request_num} response: {response_json['text']}")
        return response_json["text"]

async def main():
    black_image_bytes = create_image_bytes(color='black')
    
    print("Sending same black image 3 times...")
    await one_image_request(black_image_bytes, 1)
    await one_image_request(black_image_bytes, 2)
    await one_image_request(black_image_bytes, 3)
    print("Done! Check server logs for cache hits")

if __name__ == "__main__":
    asyncio.run(main())

