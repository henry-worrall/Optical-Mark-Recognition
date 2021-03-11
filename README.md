# Optical Mark Recognition

This project aims to automatically detect whether a mark has been made in a certain region of a scanned image based off a template. For the example given in this project, I have used a checkbox survey. The programme compares a template PNG image (converted from word document), and a JPEG image capture on my phone over the regions of interest (ROIs) which have been determined from the template image.

## Installation 

To install the software, simply clone this repository to your computer and run the `pip install -r requirement.txt` command from your local project folder.

## Project Structure

The project is separated into two python scripts. Running the OMR_main.py will run the programme, thereby comparing the template survey and the completed survey found in the image folder. OMR_main.py uses the functions in utlis.py to improve readability in the main script. Once successfully run, Python will save a csv file containing the responses to each question and display an output of how the programme has compared the images.

## Programme Algorithm

1. Define the ROIs from the template document (this can be done in Paint).
2. Load the template and filled survey.
3. Align the images using the function in utils.py.
4. Convert to  Canny image (see [here](https://en.wikipedia.org/wiki/Canny_edge_detector) for more information).
5. Compare the similarity of the ROIs between templated and completed surveys.
6. Assume that a mark has been made if the structural similarity of the ROIIs is less than 0.5.
7. Check that only one result mark is made per question.
8. Save results to a csv file.
9. Show the how the images have been compared.