import asyncio
import base64
import io
import time

import aiohttp
from PIL import Image

IMAGE_TOKEN_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def create_image_bytes(color, width=512, height=512, format="PNG"):
    img = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def one_image_request(image_data):
    start = time.time()
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
        await response.json()
    elapsed = time.time() - start
    return elapsed


async def two_images_request(image_data_1, image_data_2):
    start = time.time()
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Hello {IMAGE_TOKEN_TEXT} {IMAGE_TOKEN_TEXT}",
                "image_data": [image_data_1, image_data_2],
                "sampling_params": {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "stop_token_ids": [1],
                },
            },
        )
        await response.json()
    elapsed = time.time() - start
    return elapsed


async def main():
    black_image_bytes = create_image_bytes(color='black')
    white_image_bytes = create_image_bytes(color='white')
    
    print("Test 1: One black image (cold)")
    time1 = await one_image_request(black_image_bytes)
    print(f"  Time: {time1:.3f}s")
    
    print("\nTest 2: Two images - black (should be cached) + white (cold)")
    time2 = await two_images_request(black_image_bytes, white_image_bytes)
    print(f"  Time: {time2:.3f}s")
    
    print("\nTest 3: One black image again (should be fully cached)")
    time3 = await one_image_request(black_image_bytes)
    print(f"  Time: {time3:.3f}s")
    
    print("\nTest 4: Two images again - black + white (both should be cached)")
    time4 = await two_images_request(black_image_bytes, white_image_bytes)
    print(f"  Time: {time4:.3f}s")
    
    print("\n=== Analysis ===")
    print(f"If multimodal embedding cache is working:")
    print(f"  - Test 2 should be faster than 2x Test 1 (black image cached)")
    print(f"  - Test 3 should be much faster (full KV cache hit)")
    print(f"  - Test 4 should be similar to Test 3 (multimodal cache + some KV cache)")
    
if __name__ == "__main__":
    asyncio.run(main())

