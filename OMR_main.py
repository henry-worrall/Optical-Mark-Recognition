import cv2
import numpy as np
import pandas as pd
import imutils
import utlis
from collections import namedtuple
from skimage.metrics import structural_similarity as ssim

# parameters
imgPath = "Images/filled_survey.jpg"
templatePath = "Images/Questionnaire Template 01.png"
imgWidth = 500


# Determine locations of marks on template
# create a named tuple which we can use to create regions of interest
# of the input document which we wish to check
Location = namedtuple("Location", ["id", "bbox"])

# bbox locations take a tuple format (x,y,w,h)
# (x,y) corresponds to top-left corner of ROI
# Questions ordered numerically for reference later

# define the locations of areas of the document we wish to OMR
LOCATIONS = [
	Location("1. Strongly Agree", (263, 332, 72, 40)),
	Location("1. Agree", (334, 332, 34, 40)),
	Location("1. Neutral", (371, 332, 41, 40)),
	Location("1. Disagree", (413, 332, 48, 40)),
	Location("1. Strongly Disagree", (463, 332, 79, 40)),
    Location("2. Strongly Agree", (263, 373, 72, 40)),
	Location("2. Agree", (334, 373, 34, 40)),
	Location("2. Neutral", (371, 373, 41, 40)),
	Location("2. Disagree", (413, 373, 48, 40)),
	Location("2. Strongly Disagree", (463, 373, 79, 40)),
    Location("3. Strongly Agree", (263, 417, 72, 40)),
	Location("3. Agree", (334, 417, 34, 40)),
	Location("3. Neutral", (371, 417, 41, 40)),
	Location("3. Disagree", (413, 417, 48, 40)),
	Location("3. Strongly Disagree", (463, 417, 79, 40)),
    Location("4. Strongly Agree", (263, 459, 72, 80)),
	Location("4. Agree", (334, 459, 34, 80)),
	Location("4. Neutral", (371, 459, 41, 80)),
	Location("4. Disagree", (413, 459, 48, 80)),
	Location("4. Strongly Disagree", (463, 459, 79, 80)),
    Location("5. Strongly Agree", (263, 536, 72, 40)),
	Location("5. Agree", (334, 536, 34, 40)),
	Location("5. Neutral", (371, 536, 41, 40)),
	Location("5. Disagree", (413, 536, 48, 40)),
	Location("5. Strongly Disagree", (463, 536, 79, 40)),
    Location("6. Strongly Agree", (263, 580, 72, 40)),
	Location("6. Agree", (334, 580, 34, 40)),
	Location("6. Neutral", (371, 580, 41, 40)),
	Location("6. Disagree", (413, 580, 48, 40)),
	Location("6. Strongly Disagree", (463, 580, 79, 40)),
    Location("7. Strongly Agree", (263, 624, 72, 40)),
	Location("7. Agree", (334, 624, 34, 40)),
	Location("7. Neutral", (371, 624, 41, 40)),
	Location("7. Disagree", (413, 624, 48, 40)),
	Location("7. Strongly Disagree", (463, 624, 79, 40)),
    Location("8. Strongly Agree", (263, 669, 72, 40)),
	Location("8. Agree", (334, 669, 34, 40)),
	Location("8. Neutral", (371, 669, 41, 40)),
	Location("8. Disagree", (413, 669, 48, 40)),
	Location("8. Strongly Disagree", (463, 669, 79, 40)),
]


# load and resize images based on template width - better for alignment
img = cv2.imread(imgPath)
template = cv2.imread(templatePath)
template_width = template.shape[1]
img = imutils.resize(image=img, width=template_width)

# align images
aligned = utlis.align_images(img,template,maxFeatures=500,keepPercent=0.15)

# Convert to grayscale, and Canny threshold
alignedGray = cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
alignedBlur = cv2.GaussianBlur(alignedGray,(5,5),1)
alignedCanny = cv2.Canny(alignedBlur, 10,50)
templateGray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
templateBlur = cv2.GaussianBlur(templateGray,(5,5),1)
templateCanny = cv2.Canny(templateBlur, 10,50)

# mark boxes on alignedCanny
count = 0
answers = {}
for loc in LOCATIONS:
    count += 1
    (x, y, w, h) = loc.bbox

    # Show regions on template and image
    cv2.rectangle(template, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(aligned, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # select image for ROI on both template and image
    roi_temp_canny = templateCanny[y:y + h, x:x + w]
    roi_aligned = alignedCanny[y:y + h, x:x + w]

    # Calculate similarity between ROI on template and image
    s = ssim(roi_temp_canny, roi_aligned)

    # If box is marked, add to answers
    # Assume it is marked is similarity less than 0.5
    if s < 0.5:
        # Split id to obtain Question and Answer
        q_and_a = loc.id.split(".")
        q = q_and_a[0].strip()
        a = q_and_a[1].strip()
        if q in answers:
            answers[q].append(a)
            print(f"Recognition error! Please check question {q}.")
        else:
            answers[q] = [a]
            print(f"Answer to question {q} is {a}")
        # # Show similarity for every ROI
        # print(f"{loc.id} similarity: {s}")
    # # show first ten boxes
    # if count <= 10:
    #     cv2.imshow(str(count), roi_aligned)


# Flag recognition error or else set the answer as the only item in list
errors = False
for question in answers:
    if len(answers[question]) != 1:
        errors = True
        print(f"Error! Check question {question}")

if errors == False:
    # Write responses to csv
    responses_df = pd.DataFrame(answers)
    print(responses_df.head())
    responses_df.to_csv("responses.csv", index=False, encoding='utf-8')


# Show output
imgBlank = np.zeros_like(template)
imageArray = ([template,aligned,alignedGray,alignedBlur],
    [alignedCanny, templateCanny, imgBlank,imgBlank])

imgStacked = utlis.stackImages(imageArray, 0.5)

cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)