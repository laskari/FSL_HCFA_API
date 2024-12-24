from io import BytesIO

from src.extraction_rd_util import run_hcfa_rd_pipeline
from src.extraction_util import run_hcfa_pipeline

from fastai.learner import load_learner
from fastai.vision.all import PILImage
import torch
import pathlib
from config import *
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#IMAGE_CLASSIFIER = r"D:\project\FSL\FSL_codebase\api\HCFA\artifacts\HCFA_Classifier.pkl"


# Define Image classifier function
# hcfa_classifier = load_learner(IMAGE_CLASSIFIER, cpu=not use_cuda)
hcfa_classifier = load_learner(IMAGE_CLASSIFIER)


def grid_classifier(image_bytes: bytes, hcfa_classifier):
    """
    The function `grid_classifier` takes an image path and a classifier, makes a prediction using the
    classifier, and returns the predicted label.
    
    :param image_path: The `image_path` parameter is a string that represents the file path to the image
    that will be classified by the `hcfa_classifier`
    :type image_path: str
    :param hcfa_classifier: The `hcfa_classifier` parameter is likely an object or instance of a machine
    learning model that has a `predict` method. This method takes an image as input and returns a
    prediction, which includes the predicted label among other information
    :return: The function `grid_classifier` is returning the predicted label from the `hcfa_classifier`
    after making a prediction on the image located at the specified `image_path`.
    """
    try:
        # Convert bytes to a PIL image
        image = PILImage.create(BytesIO(image_bytes))

        prediction = hcfa_classifier.predict(item=image)
        predicted_label, _, _ = prediction
        return predicted_label
    except Exception as e:
        print(f"Error in grid_classifier {e}")
        raise e


def run_final_hcfa_pipeline(content, file_name:str):
    """
    The `run_final_hcfa_pipeline` function classifies an image as grid or non-grid and runs the
    corresponding pipeline for processing HCFA forms.
    
    :param image_path: The `run_final_hcfa_pipeline` function takes an `image_path` parameter, which is
    a string representing the file path to an image. The function then classifies the image as either
    having a grid or not, and based on this classification, it runs different pipelines to process the
    image data
    :type image_path: str
    :return: The `run_final_hcfa_pipeline` function returns the `result` and `error` variables.
    """
    try:
        # Classiffy image into no_grid or grid
        predicted_label = grid_classifier(image_bytes=content, hcfa_classifier=hcfa_classifier)

        if predicted_label == 'grid':
           print("Running the Grid Pipeline")
           result, error = run_hcfa_pipeline(content = content, image_path=file_name)
        else:
           print("Running the Non Grid Pipeline")
           result, error = run_hcfa_rd_pipeline(content = content, image_path=file_name)
        
        return result, error
    except Exception as e:
        print(f"Error while running run_final_hcfa_pipeline {e}")
        raise e
