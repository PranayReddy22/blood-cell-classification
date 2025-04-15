import xml.etree.ElementTree as ET
import cv2
import csv

image_path = "/Users/tylerrowe/Desktop/College/24Fall/Applied ML/BCCD_Dataset/BCCD"
save_path = "/Users/tylerrowe/Desktop/College/24Fall/Applied ML/AML-ClassifyingBloodImages/images"
num_annotations = 0

# Create a list to store bounding boxes and labels
annotations = []

# 410 images
for i in range(1, 411):
    # Check and see if the image exists
    if i < 10:
        image_name = "BloodImage_0000" + str(i)
    elif i < 100:
        image_name = "BloodImage_000" + str(i)
    else:
        image_name = "BloodImage_00" + str(i)

    try:
        image = cv2.imread(f"{image_path}/JPEGImages/{image_name}.jpg")
        tree = ET.parse(f"{image_path}/Annotations/{image_name}.xml")
        num_annotations += 1
    except:
        continue

    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    label = attr.text
                if 'bndbox' in attr.tag:
                    bbox = list(attr)
                    xmin = int(round(float(bbox[0].text)))
                    ymin = int(round(float(bbox[1].text)))
                    xmax = int(round(float(bbox[2].text)))
                    ymax = int(round(float(bbox[3].text)))
                    
                    # Append bounding box and label to the list
                    annotations.append({
                        "image_name": image_name,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "label": label
                    })

                    # Draw bounding box and label on the image
                    if label[0] == "R":
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                        # cv2.putText(image, label, (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
                    if label[0] == "W":
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                        # cv2.putText(image, label, (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
                    if label[0] == "P":
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                        # cv2.putText(image, label, (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)

    cv2.imwrite(f"{save_path}/annotatedImagesNoLabel/" + image_name + ".jpg", image)

# Save the annotations to a CSV file
with open("bounding_box_annotations.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "xmin", "ymin", "xmax", "ymax", "label"])
    for annotation in annotations:
        writer.writerow([annotation["image_name"], annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3], annotation["label"]])

print("Number of images annotated: ", num_annotations) # 363 images

import pandas as pd

# Load the bounding box annotations from the CSV file
df = pd.read_csv(f"{save_path}/bounding_box_annotations.csv")

# one hot encode the labels
one_hot = pd.get_dummies(df["label"])

# drop the label column
df = df.drop("label", axis=1)

# join the one hot encoded columns
df = df.join(one_hot)

df.to_csv(f"{image_path}/one_hot_bounding_box_annotations.csv", index=False)

# print number of bounding boxes
print("Number of bounding boxes: ", len(df)) # 4868 bounding boxes