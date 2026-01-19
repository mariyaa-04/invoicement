import cv2
import numpy as np
import os

print("🎨 Creating sample tractor invoice...")

# Create folder if it doesn't exist
os.makedirs("data/images", exist_ok=True)

# Create white image (500 height, 800 width, 3 channels for RGB)
img = np.ones((500, 800, 3), dtype=np.uint8) * 255

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "TRACTOR INVOICE", (50, 50), font, 1, (0, 0, 0), 2)
cv2.putText(img, "Dealer: ABC Tractors Pvt Ltd", (50, 100), font, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Model: Mahindra 575 DI", (50, 140), font, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Horse Power: 50 HP", (50, 180), font, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Asset Cost: Rs. 5,25,000", (50, 220), font, 0.7, (0, 0, 0), 2)

# Save the image
output_path = "data/images/sample_invoice.png"
cv2.imwrite(output_path, img)

print(f"✅ Created: {output_path}")
print(f"📏 Size: 800x500 pixels")
print("\n🎯 Next, run:")
print("python src/executable.py data/images/sample_invoice.png")
