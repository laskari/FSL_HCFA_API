from src.extraction_rd_util import run_hcfa_rd_pipeline
from src.extraction_util import run_hcfa_pipeline

from fastai.learner import load_learner
import torch
import pathlib
from config import *
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#IMAGE_CLASSIFIER = r"D:\project\FSL\FSL_codebase\api\HCFA\artifacts\HCFA_Classifier.pkl"


# Define Image classifier function
# hcfa_classifier = load_learner(IMAGE_CLASSIFIER, cpu=not use_cuda)
hcfa_classifier = load_learner(IMAGE_CLASSIFIER)


def grid_classifier(image_path: str, hcfa_classifier):
  try:
      prediction = hcfa_classifier.predict(item=image_path)
      predicted_label, _, _ = prediction
      return predicted_label
  except Exception as e:
    print(f"Error in grid_classifier {e}")
    raise e



def run_final_hcfa_pipeline(image_path:str):
    try:
        # Classiffy image into no_grid or grid
        predicted_label = grid_classifier(image_path=image_path, hcfa_classifier=hcfa_classifier)

        if predicted_label == 'grid':
           print("Running the Grid Pipeline")
           result, error = run_hcfa_pipeline(image_path=image_path)
        else:
           print("Running the Non Grid Pipeline")
           result, error = run_hcfa_rd_pipeline(image_path=image_path)
        
        return result, error
    except Exception as e:
        print(f"Error while running run_final_hcfa_pipeline {e}")
        raise e
