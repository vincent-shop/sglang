import asyncio
import base64
import io

import aiohttp
from PIL import Image

IMAGE_TOKEN_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def create_image_bytes(color, width=512, height=512, format="PNG"):
    """Create an image of the specified color and return its bytes as base64 string."""
    img = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def single_image_request(image_data, name="image"):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Describe this {name}: {IMAGE_TOKEN_TEXT}",
                "image_data": image_data,
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.0,
                    "stop_token_ids": [1],
                },
            },
        )
        response_json = await response.json()
        return response_json.get("text", "")

async def two_images_request(image_data_1, image_data_2):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Compare: {IMAGE_TOKEN_TEXT} and {IMAGE_TOKEN_TEXT}",
                "image_data": [image_data_1, image_data_2],
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.0,
                    "stop_token_ids": [1],
                },
            },
        )
        response_json = await response.json()
        return response_json.get("text", "")

async def main():
    print("=== Testing Qwen2.5-VL with Improved Per-Item Image Caching ===\n")
    
    # Create test images
    black_image = create_image_bytes(color='black')
    white_image = create_image_bytes(color='white')
    red_image = create_image_bytes(color='red')
    
    print("Test 1: Single BLACK image (no cache expected)")
    await single_image_request(black_image, "black image")
    
    print("\nTest 2: BLACK + WHITE images (black should cache)")
    await two_images_request(black_image, white_image)
    
    print("\nTest 3: Single WHITE image (white should cache)")
    await single_image_request(white_image, "white image")
    
    print("\nTest 4: BLACK + RED images (black should cache)")
    await two_images_request(black_image, red_image)
    
    print("\nTest 5: WHITE + RED images (both should cache)")
    await two_images_request(white_image, red_image)
    
    print("\n=== Check server logs for cache hits! ===")
    print("With the new Qwen processor changes, you should see:")
    print("- Test 1: ~332 new tokens, 0 cached")
    print("- Test 2: ~332 new (white), ~327 cached (black)")
    print("- Test 3: Few new, ~327 cached (white)")
    print("- Test 4: ~332 new (red), ~327 cached (black)")
    print("- Test 5: Few new, ~654 cached (both images)")


if __name__ == "__main__":
    asyncio.run(main())
