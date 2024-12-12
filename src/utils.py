import pandas as pd

import os
from datetime import datetime
from collections import defaultdict

def merge_donut_outputs(donut_out_old, donut_out_new, keys_from_old):
    """
    Combines the new donut output with values for certain keys taken from the old donut output.

    Parameters:
    donut_out_old (pd.DataFrame): The old donut output, format: ['Key', 'Value'].
    donut_out_new (pd.DataFrame): The new donut output, format: ['Key', 'Value'].
    keys_from_old (list): A list of keys to consider from the old donut output.

    Returns:
    pd.DataFrame: Combined DataFrame with format ['Key', 'Value'].
    """
    print("New Model Output --->>>>", donut_out_new[donut_out_new['Key'].isin(keys_from_old)])
    print("Old Model Output --->>>>", donut_out_old[donut_out_old['Key'].isin(keys_from_old)])

    # Create a dictionary of values from old output for specified keys
    old_values_for_keys = donut_out_old.set_index('Key').loc[keys_from_old, 'Value'].to_dict()

    # Update the new DataFrame with values from the old DataFrame for specified keys
    donut_out_new['Value'] = donut_out_new.apply(
        lambda row: old_values_for_keys.get(row['Key'], row['Value']), axis=1
    )

    # Return the combined DataFrame with updated values
    return donut_out_new[['Key', 'Value']]

def merge_key_aggregated_scores(scores_old, scores_new, keys_from_old):
    """
    Merges two key aggregated scores dictionaries, updating values for specified keys from the old scores.

    Parameters:
    scores_old (defaultdict(float)): The old key aggregated scores.
    scores_new (defaultdict(float)): The new key aggregated scores.
    keys_from_old (list): A list of keys to retain values from the old scores.

    Returns:
    defaultdict(float): Merged key aggregated scores.
    """
    # Create a copy of the new scores to avoid modifying the original
    merged_scores = defaultdict(float, scores_new)
    
    # Update values for specified keys from the old scores
    for key in keys_from_old:
        if key in scores_old:
            merged_scores[key] = scores_old[key]

    return merged_scores

def add_missing_keys(donut_out_old: pd.DataFrame, key_mapping: pd.DataFrame) -> pd.DataFrame:

    """
    This function checks for missing keys from 'Modified_key' in 'donut_out_old' 
    and adds them with value "[BLANK]".

    Parameters:
    - donut_out_old: DataFrame with columns ['Key', 'Value'].
    - key_mapping: DataFrame with columns ['Key_Name', 'Modified_key'].

    Returns:
    - Updated 'donut_out_old' DataFrame with missing keys added.
    """

    # Extract the keys from donut_out_old and key_mapping
    existing_keys = set(donut_out_old['Key'])
    all_keys = set(key_mapping['Key_Name'])
    
    # Find keys that are in Modified_key but missing from donut_out_old
    missing_keys = all_keys - existing_keys
    
    # Create a dataframe with missing keys and value as "[BLANK]"
    missing_entries = pd.DataFrame({'Key': list(missing_keys), 'Value': "[BLANK]"})
    
    # Concatenate the original dataframe with the missing entries
    updated_donut_out_old = pd.concat([donut_out_old, missing_entries], ignore_index=True)
    
    return updated_donut_out_old

def log_message(log_file: str, message: str):
    """
    Logs a message to a .txt file, creating the file if it doesn't exist.

    Args:
    - log_file (str): Path to the log file.
    - message (str): The message to log.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get the current timestamp for the log entry
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Open the file in append mode and write the message
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")