import base64
import os
from io import BytesIO

import requests
from PIL import Image

# Create a dummy image for testing
if not os.path.exists("samples/images"):
    os.makedirs("samples/images")
Image.new("RGB", (128, 128), color="red").save("samples/images/test_image_1.png")
Image.new("RGB", (64, 64), color="blue").save("samples/images/test_image_2.png")


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def main():
    api_url = "http://localhost:8080"

    # 1. Process images
    image_paths = [
        "samples/images/test_image_1.png",
        "samples/images/test_image_2.png",
    ]
    encoded_images = [image_to_base64(p) for p in image_paths]
    response = requests.post(
        f"{api_url}/process/images", json={"images": encoded_images}
    )
    response.raise_for_status()
    image_embeddings = response.json()["embeddings"]
    print("Successfully processed images.")

    # 2. Process queries
    queries = [
        "What is the capital of France?",
        "Explain quantum computing.",
    ]
    response = requests.post(f"{api_url}/process/queries", json={"queries": queries})
    response.raise_for_status()
    query_embeddings = response.json()["embeddings"]
    print("Successfully processed queries.")

    # 3. Score multi-vector
    response = requests.post(
        f"{api_url}/process/score_multi_vector",
        json={
            "image_embeddings": image_embeddings,
            "query_embeddings": query_embeddings,
        },
    )
    response.raise_for_status()
    scores = response.json()["scores"]
    print("Successfully scored multi-vector.")
    print("Scores:", scores)


if __name__ == "__main__":
    main()
