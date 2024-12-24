import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form, HTTPException

from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form
import torch
import json, os, sys

# from extraction_util import run_hcfa_pipeline
from src.pipeline import run_final_hcfa_pipeline
# from src.utils import log_message
from src.logger import log_message, setup_logger
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

logger = setup_logger(LOGFILE_DIR)

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/hcfa_extraction")
async def ml_extraction(data: dict):
    try:
        # Log start of the extraction
        log_message(logger, "Started ml_extraction", level="INFO")

        # Get the image path from the payload
        image_file_path = data.get('FilePath')

        if not image_file_path:
            log_message(logger, "FilePath field is required", level="ERROR")
            raise HTTPException(status_code=400, detail="FilePath field is required")

        if not os.path.exists(image_file_path):
            log_message(logger, f"File not found: {image_file_path}", level="ERROR")
            raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

        log_message(logger, f"File found: {image_file_path}. Running pipeline...", level="INFO")

        # Run the pipeline and capture result and error
        result, error = run_final_hcfa_pipeline(file_name = image_file_path)

        if error:
            # Log the error before raising HTTPException
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        log_message(logger, "Pipeline ran successfully", level="INFO")

        # If there's no error, return the result with file path
        response_data = {"version" : VERSION, "file_path": data.get('FilePath'), "result": result['result']}
        
        # Log the successful result
        log_message(logger, f"Extraction result: {response_data}", level="INFO")
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )

@app.post("/hcfa_extraction_streaming")
async def ml_extraction(file: UploadFile = File(...)):
    try:
        # Log start of the extraction
        log_message(logger, "Started ml_extraction", level="INFO")
        # Read the contents of the uploaded file
        contents = await file.read()

        # Get file name
        file_name = file.filename

        # Get the image path from the payload
        # image_file_path = data.get('FilePath')

        # if not image_file_path:
        #     log_message(logger, "FilePath field is required", level="ERROR")
        #     raise HTTPException(status_code=400, detail="FilePath field is required")

        # if not os.path.exists(image_file_path):
        #     log_message(logger, f"File not found: {image_file_path}", level="ERROR")
        #     raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

        # log_message(logger, f"File found: {image_file_path}. Running pipeline...", level="INFO")

        # Run the pipeline and capture result and error
        result, error = run_final_hcfa_pipeline(contents)

        if error:
            # Log the error before raising HTTPException
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        log_message(logger, "Pipeline ran successfully", level="INFO")

        # If there's no error, return the result with file path
        response_data = {"version" : VERSION, "file_name": file_name, "result": result['result']}
        
        # Log the successful result
        log_message(logger, f"Extraction result: {response_data}", level="INFO")
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )



if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

