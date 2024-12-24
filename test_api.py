import requests
import time

# Define the URL of the API endpoint
url = "http://localhost:8000/hcfa_extraction"

# Define the payload data as a dictionary
data = {
    # "FilePath": r"/Data/FSL_codebase/FSL_HCFA_API/artifacts/1149142YAZ006_001.png"
    "FilePath": r"D:\project\FSL\new_codebase\FSL_HCFA_API\artifacts\HER9Q42WO001_001.tiff"

}

# try:
#     # Measure the start time
#     start_time = time.time()

#     # Open the file in binary mode
#     with open(data['FilePath'], "rb") as file:
#         # Prepare the file to be uploaded
#         files = {"file": file}

#         # Send the POST request with the file
#         response = requests.post(url, files=files)

#     # Print the response
#     print(response.json())
#     total_time_taken = time.time() - start_time

#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         # Print the response content
#         print("Response:", response.json())
#         print("Request was successful.")
#         print(f"Total time taken for execution: {total_time_taken}")
#     else:
#         # If request was not successful, print error message
#         print("Error:", response.content)
#         print(f"Request failed with status code {response.status_code}")
# except Exception as e:
#     print("Error occurred during API call:", e)

try:
    # Measure the start time
    start_time = time.time()

    # Make the POST request with the JSON payload
    response = requests.post(url, json=data)

    # Measure the total time taken for execution
    total_time_taken = time.time() - start_time

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        print("Response:", response.json())
        print("Request was successful.")
        print(f"Total time taken for execution: {total_time_taken}")

    else:
        # If request was not successful, print error message
        print("Error:", response.content)
        print(f"Request failed with status code {response.status_code}")
except Exception as e:
    print("Error occurred during API call:", e)