import uvicorn, os
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, HTMLResponse

app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Log file paths for different services

### TODO -: Change it with the service name like HCFA, UB and ADA. Dame changes needs to be done in frontend/test.html
log_files = {
    "service1": os.path.join(os.getcwd(), "logs", "hcfa_logs.log"),
    "service2": os.path.join(os.getcwd(), "logs", "hcfa_logs copy.log"),
    "service3": os.path.join(os.getcwd(), "logs", "ub.log")
}

# Optional: Redirect root to the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=open("frontend/test.html").read())

@app.get("/read-error-log")
async def read_error_log(service: str = Query(..., description="Service name")):
    log_file = log_files.get(service)
    if not log_file:
        raise HTTPException(status_code=404, detail="Service log file not found.")
    try:
        with open(log_file, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    # port = int(sys.argv[1]) if  len(sys.argv) > 1 else 5000
    uvicorn.run("tracking:app", host="0.0.0.0", port=5000)