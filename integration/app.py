# app.py
from fastapi import FastAPI, File, UploadFile
import pydicom
from io import BytesIO
from typing import List

app = FastAPI()

@app.post("/upload-images/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        # Read the file using pydicom
        dicom_data = pydicom.dcmread(BytesIO(await file.read()))
        
        # Process the DICOM data (e.g., segmentation)
        # This would be where you call your segmentation model
        # For now, we'll just print out the Patient ID for demo purposes
        print(f"Received image for Patient ID: {dicom_data.PatientID}")
        
        # Perform real-time segmentation and other processing here
        # For the purposes of this example, we will just pass
        # Instead, you would call your model and return the segmentation results
        
        # Save the processed file or send it to another service
        # Here, we would save or forward the results
        
    return {"filenames": [file.filename for file in files]}

# This would run the server when the script is executed directly
# Use the command: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
