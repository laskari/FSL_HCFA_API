import json
import torch
import io
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

from config import *

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
average_coordinates_hcfa_df = pd.read_excel(HCFA_AVERAGE_COORDINATE_PATH)
key_mapping = pd.read_excel(HCFA_FORM_KEY_MAPPING)
mapping_dict = key_mapping.set_index('Key_Name').to_dict()['Modified_key']
reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}


class HCFARoiPredictor:
    def __init__(self, model_path, category_mapping_path=CATEGORY_MAPPING_PATH):
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
            return {c['id'] + 1: c['name'] for c in json.load(f)['categories']}

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
        # APPLYING Post prrocessing
        # Calculate mean values for specific labels
        xmin_24_table = infer_df.loc[infer_df['class_name'] == '9. d. INSURANCE PLAN NAME OR PROGRAM NAME', 'x0'].mean()
        xmax_patient_birth_date = infer_df.loc[infer_df['class_name'] == '3. PATIENT’S BIRTH DATE', 'x1'].mean()
        xmax_21_diagnosis_or_nature_of_illness = infer_df.loc[infer_df['class_name'] == '21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY.', 'x1'].mean()
        # 1a. INSURED’S I.D. NUMBER
        xmax_1a_insured_id_number = infer_df.loc[infer_df['class_name'] == '1a. INSURED’S I.D. NUMBER', 'x1'].mean()


        # Apply post-processing for '1_InsType' class
        infer_df.loc[(infer_df['class_name'] == '1_InsType'), 'x0'] = xmin_24_table
        infer_df.loc[(infer_df['class_name'] == '1_InsType'), 'x1'] = xmax_patient_birth_date

        # Apply post-processing for '35_Remarks' class
        infer_df.loc[(infer_df['class_name'] == '12_Patient_Auth_Sign'), 'x0'] = xmin_24_table
        infer_df.loc[(infer_df['class_name'] == '12_Patient_Auth_Sign'), 'x1'] = xmax_patient_birth_date

        infer_df.loc[(infer_df['class_name'] == '21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY.'), 'x0'] = xmin_24_table
        infer_df.loc[(infer_df['class_name'] == '21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY.'), 'x1'] = xmax_patient_birth_date

        infer_df.loc[(infer_df['class_name'] == '19. Additional Claim Information'), 'x0'] = xmin_24_table
        infer_df.loc[(infer_df['class_name'] == '19. Additional Claim Information'), 'x1'] = xmax_patient_birth_date

        # 24. Table
        infer_df.loc[(infer_df['class_name'] == '24. Table'), 'x0'] = xmin_24_table
        infer_df.loc[(infer_df['class_name'] == '24. Table'), 'x1'] = xmax_1a_insured_id_number
        return infer_df

    def predict_image(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)
        # pil_image = Image.open(image_path)
        # to_tensor = transforms.ToTensor()
        # image = to_tensor(pil_image)
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
        file_name = [image_path.split(".")[0]] * num_predictions
        # file_name = [image_path] * num_predictions


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

        post_processed_df = self._postprocessing_annotation(infer_df)
        return post_processed_df

# Load the RPI model
frcnn_predictor = HCFARoiPredictor(MODEL_PATH)


def roi_model_inference(image_path, image):
    result_df = frcnn_predictor.predict_and_get_dataframe(image_path, image)
    max_score_indices = result_df.groupby('class_name')['score'].idxmax()
    result_df = result_df.loc[max_score_indices]
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
        return_dict_in_generate=True,
    )
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace("<one>", "1")
    prediction = processor.token2json(prediction)
    return prediction, outputs

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
        # Laskari-Naveen/HCFA_102 Laskari-Naveen/HCFA_99
        processor_1 = AutoProcessor.from_pretrained("Laskari-Naveen/HCFA_99")
        model_1 = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/HCFA_99")
        model_1.eval().to(device)

        processor_2 = AutoProcessor.from_pretrained("Laskari-Naveen/HCFA_102")
        model_2 = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/HCFA_102")
        model_2.eval().to(device)
        print("Model loaded successfully")
    except:
        print("Model Loading failed !!!")
    return processor_1, model_1, processor_2, model_2 


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

def map_result2(dict1, dict2):
    result_dict_2 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_2[key] = {
                    "Mapping_key": mapping_keys,
                    "coordinates": value
                }
    return result_dict_2

def map_result1_final_output(result_dict_1, additional_info_dict):
    updated_result_dict_1 = {}

    # Iterate over additional_info_dict
    for key, additional_info in additional_info_dict.items():
        # Check if the key exists in result_dict_1
        if key in result_dict_1:
            coordinates = result_dict_1[key]
        else:
            # If the key is missing in result_dict_1, set coordinates to None
            coordinates = None

        # Store the coordinates and additional_info in updated_result_dict_1
        updated_result_dict_1[key] = {"coordinates": coordinates, "text": additional_info}

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


def run_hcfa_pipeline(image_path: str):
    try:
        # image_path = os.path.join(input_image_folder, each_image)
        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = Image.open(io.BytesIO(image_path)).convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(pil_image)

        """USE TWO MODEL TO HANDLE THE BLANK KEY 
            - model_1 (old model)
            - model_2 (new model)
            - processor_1 (new processor)
            - processor_2 (old processor)

            convert_hcfa_predictions_to_df_old and
            convert_hcfa_predictions_to_df_new

        """
        prediction_old, output = run_prediction_donut(pil_image, model_1, processor_1)
        donut_out_old = convert_hcfa_predictions_to_df(prediction_old, version = "old")

        from src.utils import merge_donut_outputs, add_missing_keys
        
        ###### CHECK IF ANY KEY MISSING ######
        donut_out_old = add_missing_keys(donut_out_old, key_mapping)

        prediction_new, output = run_prediction_donut(pil_image, model_2, processor_2)
        donut_out_new = convert_hcfa_predictions_to_df(prediction_new, version = "new")
        
        ###### MERGE OLD AND NEW MODEL OUTPUT #######
        donut_out = merge_donut_outputs(donut_out_old, donut_out_new, KEYS_FROM_OLD)

        # What is this? Is it Mapping the donut keys to XML values? Can't understand.
        # for old_key, new_key in reverse_mapping_dict.items():
        #     donut_out["Key"].replace(old_key, new_key, inplace=True)

        # This is just converting the dataframe to dictionary
        json_data = donut_out.to_json(orient='records')
        data_list = json.loads(json_data)
        # output_dict_donut = {item['Key']: item['Value'] for item in data_list}

        output_dict_donut = {}

        # Iterate through the data_list
        for item in data_list:
            key = item['Key']
            value = item['Value']

            # Check if the key already exists in the output dictionary
            if key in output_dict_donut:
                # If the key exists, append the value to the list of dictionaries
                output_dict_donut[key].append({'value': value})
            else:
                # If the key doesn't exist, create a new list with the current value
                output_dict_donut[key] = [{'value': value}]


        print("Length of Keys being outputed", len(output_dict_donut.keys()))
        # This is just doing the ROI inference and converting DF to dict
        res = roi_model_inference(image_path, image)
        df_dict = res.to_dict(orient='records')

        # Implementing the average part here

        # Convert the average coordinates DataFrame to a dictionary for easy access
        average_coordinates_dict = average_coordinates_hcfa_df.set_index('label').to_dict(orient='index')

        # Get all unique class names
        all_class_names = set(average_coordinates_hcfa_df['label'])

        # Initialize the output dictionary
        output_dict_det = {}

        # Iterate over all class names
        for class_name in all_class_names:
            # Check if the class name exists in df_dict
            item = next((item for item in df_dict if item['class_name'] == class_name), None)
            if item:
                # If the class name exists, use the coordinates from df_dict
                x1, y1, x2, y2 = item['x0'], item['y0'], item['x1'], item['y1']
            else:
                # If the class name doesn't exist, replace coordinates with average coordinates
                avg_coords = average_coordinates_dict.get(class_name, None)
                if avg_coords:
                    x1 = avg_coords['xmin']
                    y1 = avg_coords['ymin']
                    x2 = avg_coords['xmax']
                    y2 = avg_coords['ymax']
                else:
                    # If average coordinates are not available, set coordinates to NaN
                    x1, y1, x2, y2 = float('nan'), float('nan'), float('nan'), float('nan')

            # Store the coordinates in the output dictionary
            output_dict_det[class_name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        # Map the ROI keys with the Donut keys
        result_dict_1 = map_result1(output_dict_det, BBOX_HCFA_DONUT_Mapping_Dict)
        # result_dict_2 = map_result2(output_dict_det, BBOX_DONUT_Mapping_Dict)
        final_mapping_dict  = map_result1_final_output(result_dict_1, output_dict_donut)

        return {"result": final_mapping_dict}, None
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application description")
    parser.add_argument("input_image_folder", help="Path to the input image folder")
    parser.add_argument("output_ROI_folder", help="Path to the output ROI folder")
    parser.add_argument("output_extraction_folder", help="Path to the output extraction folder")
    args = parser.parse_args()
    processor, model = load_model(device)
    # run_application(args.input_image_folder, args.output_ROI_folder, args.output_extraction_folder)