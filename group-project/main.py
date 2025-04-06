import cv2
import pytesseract
import re
import matplotlib.pyplot as plt
import textwrap
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"D:\apps\tesseract.exe"

image = cv2.imread("group-project//whywasthiswoman.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=custom_config)

pattern = r"(.+?)\s+(\d+)$"
extracted_data = re.findall(pattern, text, re.MULTILINE)
categories = [item[0].strip() for item in extracted_data]
percentages = [int(item[1]) for item in extracted_data]

def wrap_labels(labels, width=30):
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]

wrapped_categories = wrap_labels(categories)

plt.figure(figsize=(12, 7))
colors = plt.cm.Paired(np.linspace(0, 1, len(categories)))

bars = plt.barh(wrapped_categories, percentages, color=colors, edgecolor="black")

for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_width() +0.5, bar.get_y() + bar.get_height()/2, 
                f"{percentage}%", va='center', fontsize=10, fontweight='bold')

plt.xlabel("Percentage (%)", fontsize=14, fontweight="bold")
plt.text(plt.xlim()[0]-5, len(categories) - 8.5, "Reasons", 
         fontsize=14, fontweight="bold", ha="left", va="bottom")
plt.title("Why Was This Woman Kidnapped?", fontsize=16, fontweight="bold")
plt.gca().invert_yaxis()

plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout(pad=1) 
plt.show()