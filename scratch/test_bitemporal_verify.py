import requests
import io
import sys
from PIL import Image
import numpy as np

# Create two test images
img_a = Image.fromarray(np.full((200, 200, 3), [100, 150, 100], dtype=np.uint8))
img_b = Image.fromarray(np.full((200, 200, 3), [180, 120, 120], dtype=np.uint8))

buf_a = io.BytesIO()
img_a.save(buf_a, format="PNG")
buf_a.seek(0)

buf_b = io.BytesIO()
img_b.save(buf_b, format="PNG")
buf_b.seek(0)

files = {
    "image_a": ("before.png", buf_a, "image/png"),
    "image_b": ("after.png", buf_b, "image/png"),
}
data = {
    "query": "Compare Image A and Image B and show changes",
}

res = requests.post("http://127.0.0.1:8000/api/analyze/change", files=files, data=data)
print("Status Code:", res.status_code)
json_data = res.json()
with open("scratch/bitemporal_result.txt", "w", encoding="utf-8") as f:
    f.write("Answer: " + str(json_data.get("answer")) + "\n\n")
    f.write("Summary: " + str(json_data.get("summary")) + "\n\n")
    f.write("Statistics: " + str(json_data.get("statistics")) + "\n")
print("Saved result to scratch/bitemporal_result.txt")
