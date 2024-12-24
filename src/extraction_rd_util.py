import json
import torch
import io
import time
import torchvision
import pandas as pd
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision import transforms
import pandas as pd
from transformers import AutoProcessor, VisionEncoderDecoderModel
import requests
import json
from PIL import Image
import torch
import argparse
# import os
import warnings
# from tqdm import tqdm

from src.logger import log_message, setup_logger
from config import *

warnings.filterwarnings('ignore')
logger = setup_logger(LOGFILE_DIR)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

average_coordinates_hcfa_df = pd.read_excel(HCFA_AVERAGE_COORDINATE_PATH)
key_mapping = pd.read_excel(HCFA_FORM_KEY_MAPPING)
mapping_dict = key_mapping.set_index('Key_Name').to_dict()['Modified_key']
reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}


import json

class HCFARDRoiPredictor:
    def __init__(self, model_path, category_mapping_path=RD_CATEGORY_MAPPING_PATH):
        self.category_mapping = self._load_category_mapping(category_mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = len(self.category_mapping) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def _load_category_mapping(self, category_mapping_path):
        with open(category_mapping_path) as f:
            return {c['id']: c['name'] for c in json.load(f)['categories']}

    def _get_transforms(self):
        return T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])

    def _apply_nms(self, orig_prediction, iou_thresh=0.3):
        keep = torchvision.ops.nms(
            orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]
        return final_prediction

    def _postprocessing_annotation(self, infer_df):
        if len(infer_df[infer_df['class_name'] == "23_class"]) == 0:
          # print(f"working on {image_path}")
          # Select rows with class 22_label
          print("Post processing class 23")
          class_22_rows = infer_df[infer_df['class_name'] == "22_class"]

          # Create a new DataFrame with label set to 23_label
          new_rows = class_22_rows.copy()
          new_rows['class_name'] = "23_class"

          # Append the new rows to the original DataFrame
          infer_df = pd.concat([infer_df, new_rows], ignore_index=True)

        return infer_df

    def predict_image(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

    def predict_and_get_dataframe(self, image_path, image,  iou_thresh=0.5):
        predictions = self.predict_image(image)
        pred = predictions[0]
        pred_nms = self._apply_nms(pred, iou_thresh=iou_thresh)

        pred_dict = {
            'boxes': pred_nms['boxes'].cpu().numpy(),
            'labels': pred_nms['labels'].cpu().numpy(),
            'scores': pred_nms['scores'].cpu().numpy()
        }

        boxes_flat = pred_dict['boxes'].reshape(-1, 4)
        labels_flat = pred_dict['labels'].reshape(-1)
        scores_flat = pred_dict['scores'].reshape(-1)

        class_names = [self.category_mapping[label_id] for label_id in labels_flat]
        num_predictions = len(boxes_flat)
        # file_name = [image_path.split(".")[0]] * num_predictions
        file_name = [image_path] * num_predictions


        infer_df = pd.DataFrame({
            'file_name': file_name,
            'x0': boxes_flat[:, 0],
            'y0': boxes_flat[:, 1],
            'x1': boxes_flat[:, 2],
            'y1': boxes_flat[:, 3],
            'label': labels_flat,
            'class_name': class_names,
            'score': scores_flat
        })

        infer_df = self._postprocessing_annotation(infer_df)
        return infer_df

# Load the RPI model
frcnn_predictor_hcfa_rd = HCFARDRoiPredictor(model_path = HCFA_RD_MODEL_PATH)


def roi_model_inference(image_path, image,):
    result_df = frcnn_predictor_hcfa_rd.predict_and_get_dataframe(image_path, image)
    max_score_indices = result_df.groupby('class_name')['score'].idxmax()
    result_df = result_df.loc[max_score_indices]
#     print("DataFrame", result_df[["class_name", "x0", "x1", "y0", "y1"]])
    result_df = result_df[["class_name", "x0", "x1", "y0", "y1"]]
    return result_df

def run_prediction_donut(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=2,
        epsilon_cutoff=6e-4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        output_scores=True,
        return_dict_in_generate=True,
    )
    scores = outputs.scores 
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace("<one>", "1")
    prediction = processor.token2json(prediction)
    return prediction, outputs, scores

def split_and_expand(row):
    if row['Key'] == "33_Missing_Teeth":
        keys = [row['Key']]
        values = row['Value'].split(';')[0]
    else:
        keys = [row['Key']] * len(row['Value'].split(';'))
        values = row['Value'].split(';')
    return pd.DataFrame({'Key': keys, 'Value': values})


def load_model(device):
    try:
        # Laskari-Naveen/hcfa_rd_v1
        processor_1 = AutoProcessor.from_pretrained("Laskari-Naveen/hcfa_rd_v1")
        model_1 = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/hcfa_rd_v1")
        model_1.eval().to(device)

        processor_2 = AutoProcessor.from_pretrained("Laskari-Naveen/HCFA_102")
        model_2 = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/HCFA_102")
        model_2.eval().to(device)
        print("Model loaded successfully")
    except:
        print("Model Loading failed !!!")
    return processor_1, model_1, processor_2, model_2 

import math
from collections import defaultdict

def calculate_key_aggregated_scores(scores, outputs, processor):
    """
    Calculate aggregated scores for each key from model outputs and scores.

    Args:
        scores (list): The list of score tensors for each decoding step.
        outputs (obj): The output object from the model containing generated sequences.
        processor (obj): The processor object for decoding tokens.

    Returns:
        dict: A dictionary with keys as the tokenized keys and their aggregated scores.
    """
    key_aggregated_scores = defaultdict(float)

    # Token IDs generated by the model (excluding input tokens like <s>)
    generated_token_ids = outputs.sequences[0][1:]  # Exclude the first token (<s>)

    current_key = None  # Track the current key during decoding
    token_scores = []  # Temporary list to store scores of intermediate tokens
    row_scores = []  # Temporary list for scores within semicolon-separated rows

    for idx, (score, token_id) in enumerate(zip(scores, generated_token_ids)):
        # Decode the token
        token = processor.tokenizer.decode([token_id.item()], skip_special_tokens=False)

        # Detect the start of a new key
        if token.startswith("<s_") and not token.startswith("</"):

            # print(f"Start of the token {token}")
            # Start a new key; reset token scores.
            # From a text remove <s_ and >
            current_key = token[3:-1]
            # current_key = token
            token_scores = []
            row_scores = []

        # Detect the end of the current key
        elif token.startswith("</") and current_key is not None:

            # print(f"End of the token {token}")
            # Compute the aggregated score for the key
            if token_scores:
                product_of_scores = math.prod(token_scores)
                aggregated_score = product_of_scores ** (1 / len(token_scores))
                row_scores.append(aggregated_score)

            # Assign row scores to the key
            key_aggregated_scores[current_key] = row_scores if len(row_scores) > 1 else row_scores[0]
            current_key = None  # Reset the key tracking
        
        # Process intermediate tokens
        elif current_key is not None:
            # Calculate the token's probability
            max_score = torch.softmax(score, dim=-1).max().item()

            # Handle row separators
            if token == ";":
                # print("Calculating Intermedeate")
                # Calculate and store the score for the current row
                if token_scores:
                    product_of_scores = math.prod(token_scores)
                    aggregated_score = product_of_scores ** (1 / len(token_scores))
                    row_scores.append(aggregated_score)
                    token_scores = []  # Reset token scores for the next row
            elif not token.startswith("<") and not token.startswith("</"):
                # Include the score for intermediate tokens
                # print(processor.tokenizer.decode([token_id.item()], skip_special_tokens=False))
                # print(max_score)
                token_scores.append(max_score)

    return key_aggregated_scores

def convert_hcfa_predictions_to_df(prediction, version = 'new'):
    expanded_df = pd.DataFrame()
    result_df_each_image = pd.DataFrame()    
    each_image_output = pd.DataFrame(list(prediction.items()), columns=["Key", "Value"])
    try:    
        expanded_df = pd.DataFrame(columns=['Key', 'Value'])
        for index, row in each_image_output[each_image_output['Value'].str.contains(';')].iterrows():
            expanded_df = pd.concat([expanded_df, pd.DataFrame(split_and_expand(row))], ignore_index=True)

        result_df_each_image = pd.concat([each_image_output, expanded_df], ignore_index=True)
        result_df_each_image = result_df_each_image.drop(result_df_each_image[result_df_each_image['Value'].str.contains(';')].index)

        if version == 'old':
            for old_key, new_key in reverse_mapping_dict.items():
                result_df_each_image["Key"].replace(old_key, new_key, inplace=True)

    except Exception as e:
        print(f"Error in convert_hcfa_predictions_to_df {e}")
        pass
        
    return result_df_each_image

# def plot_bounding_boxes(image, df, enable_title = False):
#     image = image.permute(1,2,0)
#     colors = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'brown']
#     fig, ax = plt.subplots(1, figsize=(50, 50))
#     ax.set_aspect('auto')
#     ax.imshow(image)
#     for index, row in df.iterrows():
#         class_name = row['class_name']
#         x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
#         box_color = random.choice(colors)
#         rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor=box_color, facecolor='none')
#         ax.add_patch(rect)

#         if enable_title:
#             ax.text(x0, y0, class_name, color=box_color, fontsize=9, weight='bold')
#     ax.axis('off')
#     plt.show()

def map_result1(dict1, dict2):
    result_dict_1 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_1[mapping_key] = value
    return result_dict_1


def map_result1_final_output(result_dict_1, additional_info_dict, key_aggregated_scores):
    updated_result_dict_1 = {}

    # Iterate over additional_info_dict
    for key, additional_info in additional_info_dict.items():
        # Check if the key exists in result_dict_1
        if key in result_dict_1:
            coordinates = result_dict_1[key]
        else:
            # If the key is missing in result_dict_1, set coordinates to None
            coordinates = None
        
        if key in key_aggregated_scores:
            confidence_score = key_aggregated_scores[key]
        else:
            confidence_score = None

        # Store the coordinates and additional_info in updated_result_dict_1
        updated_result_dict_1[key] = {
            "coordinates": coordinates, 
            "text": additional_info, 
            "confidence_score" : confidence_score
        }

    return updated_result_dict_1

# def run_application(input_image_folder, output_ROI_folder, output_extraction_folder):
#     root = os.getcwd()
#     os.makedirs(output_ROI_folder, exist_ok=True)
#     os.makedirs(output_extraction_folder, exist_ok=True)
#     # image_list = os.listdir(os.path.join(root, input_image_folder))
#     image_list = os.listdir(input_image_folder)
#     for each_image in tqdm(image_list):
#         image_path = os.path.join(input_image_folder, each_image)
#         pil_image = Image.open(image_path).convert('RGB')
#         to_tensor = transforms.ToTensor()
#         image = to_tensor(pil_image)
        
#         print("Staring ROI extraction")
#         # print(uploaded_file.)
#         fasterrcnn_result_df = roi_model_inference(image_path, image)
#         print("Staring data extraction")
#         prediction, output = run_prediction_donut(image, model, processor)
#         extraction_df = convert_predictions_to_df(prediction)

#         output_ROI_path = os.path.join(root, output_ROI_folder,each_image.split(".")[0]+".xlsx" )
#         fasterrcnn_result_df.to_excel(output_ROI_path, index=False)

#         output_extraction_path = os.path.join(root, output_extraction_folder, each_image.split(".")[0]+".xlsx" )
#         extraction_df.to_excel(output_extraction_path, index=False)


# Load the models
processor_1, model_1, processor_2, model_2 = load_model(device)

def run_hcfa_rd_pipeline(content, image_path: str):
    try:
        global_start_time = time.time()
        log_message(logger, "Starting HCFA RD pipeline", level="INFO")

        # Log image path
        log_message(logger, f"Received image path: {image_path}", level="INFO")

        print("got the image")
        # Load and convert the image
        # pil_image = Image.open(image_path).convert('RGB')
        pil_image = Image.open(io.BytesIO(content)).convert('RGB')
        image_height, image_width = pil_image.size[0], pil_image.size[1]
        to_tensor = transforms.ToTensor()
        image = to_tensor(pil_image)
        log_message(logger, "Image loaded and converted to tensor", level="INFO")

        # Run old model prediction
        log_message(logger, "Running prediction with old model", level="INFO")
        prediction_old, output_old, scores_old = run_prediction_donut(pil_image, model_1, processor_1)
        donut_out_old = convert_hcfa_predictions_to_df(prediction_old, version = "old")

        key_aggregated_scores_old = calculate_key_aggregated_scores(scores_old, output_old, processor_1)
        
        log_message(logger, "Old model prediction complete", level="INFO")
        log_message(logger, f"TIME TAKEN BY OLD MODEL {time.time() - global_start_time}", level="INFO")

        # Check for missing keys and update
        from src.utils import add_missing_keys
        donut_out_old = add_missing_keys(donut_out_old, key_mapping)
        log_message(logger, "Missing keys handled in old model output", level="INFO")

        # Run new model prediction
        log_message(logger, "Running prediction with new model", level="INFO")
        prediction_new, output_new, scores_new = run_prediction_donut(pil_image, model_2, processor_2)
        donut_out_new = convert_hcfa_predictions_to_df(prediction_new, version = "new")

        key_aggregated_scores_new = calculate_key_aggregated_scores(scores_new, output_new, processor_2)
        log_message(logger, "New model prediction complete", level="INFO")
        log_message(logger, f"TIME TAKEN BY NEW MODEL {time.time() - global_start_time}", level="INFO")

        # Merge outputs from old and new models
        from src.utils import merge_donut_outputs, merge_key_aggregated_scores
        donut_out = merge_donut_outputs(donut_out_old, donut_out_new, KEYS_FROM_OLD)
        key_aggregated_scores = merge_key_aggregated_scores(key_aggregated_scores_old, key_aggregated_scores_new, KEYS_FROM_OLD)

        log_message(logger, "Merged old and new model outputs", level="INFO")

        # Convert merged output to dictionary
        json_data = donut_out.to_json(orient='records')
        data_list = json.loads(json_data)
        output_dict_donut = {}

        # Iterate through the data_list to create output_dict_donut
        for item in data_list:
            key = item['Key']
            value = item['Value'].strip()
            if key in output_dict_donut:
                output_dict_donut[key].append({'value': value})
            else:
                output_dict_donut[key] = [{'value': value}]
        
        log_message(logger, "Donut output processed into dictionary", level="INFO")

        # Start ROI inference and convert results to dictionary
        log_message(logger, "Starting ROI inference", level="INFO")
        res = roi_model_inference(image_path, image)
        df_dict = res.to_dict(orient='records')

        output_dict_det = {}
        for item in df_dict:
            class_name = item['class_name']
            x1, y1, x2, y2 = item['x0'], item['y0'], item['x1'], item['y1']
            output_dict_det[class_name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        
        log_message(logger, "ROI inference complete", level="INFO")

        # Map the ROI keys with the Donut keys
        log_message(logger, "Mapping ROI keys with Donut keys", level="INFO")
        result_dict_1 = map_result1(output_dict_det, group_key_mapping_dict)

        log_message(logger, "Processing overlaps", level="INFO")
        result_dict_2 = map_result1(result_dict_1, mapping_overlaps)

        # Add any required missing keys
        apply_19_key_post_processing = ['Box19A_Provider', '19B_ProvCredential', '19B_ProvSuffix', 'Box19B_Provider', '19B_ProvLName', '19A_ProvMI', '19A_ProvSuffix', '19A_ProvPrefix', 'Box19B_NPI', '19A_ProvLName', '19B_ProvMI', '19B_ProvFName', '19B_ProvFullNameQual', '19A_ProvFName', '19B_ProvPrefix', '19A_ProvFullNameQual', 'Box19A_QQ', "Box19B_QQ'", '19A_ProvCredential', 'Box19A_NPI']
        apply_8_pat_key_post_processing = ["8_PatStatus", "8_PatStudent"]

        for missing_keys in apply_19_key_post_processing:
            result_dict_2[missing_keys] = {'x1': 1, 'y1': 1, 'x2': 100, 'y2': 100}

        for missing_keys in apply_8_pat_key_post_processing:
            result_dict_2[missing_keys] = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}

        result_dict_2["32_MedicaidTaxId"] = {'x1': 1, 'y1': 1, 'x2': image_height, 'y2': image_width}

        # Combine the results
        result_dict_2.update(result_dict_1)
        final_mapping_dict = map_result1_final_output(result_dict_2, output_dict_donut, key_aggregated_scores)

        log_message(logger, "Pipeline processing complete", level="INFO")
        log_message(logger, f"TIME TAKEN BY COMPLETE PIPELINE {time.time() - global_start_time}", level="INFO")
        return {"result": final_mapping_dict}, None

    except Exception as e:
        log_message(logger, f"Error in run_hcfa_rd_pipeline: {e}", level="ERROR")
        return None, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application description")
    parser.add_argument("input_image_folder", help="Path to the input image folder")
    parser.add_argument("output_ROI_folder", help="Path to the output ROI folder")
    parser.add_argument("output_extraction_folder", help="Path to the output extraction folder")
    args = parser.parse_args()
    processor, model = load_model(device)
    # run_application(args.input_image_folder, args.output_ROI_folder, args.output_extraction_folder)