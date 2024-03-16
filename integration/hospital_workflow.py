# integration/hospital_workflow.py
import requests

# Constants representing the API endpoints
HOSPITAL_API_BASE_URL = 'https://hospital.api'
UPDATE_RECORD_ENDPOINT = '/update_patient_record'

# Your API key or token for authentication (keep it secure!)
API_KEY = 'your_api_key'

def update_patient_record(patient_id, analysis_data):
    """
    Update the patient's medical record with new analysis data.

    Parameters:
    - patient_id (str): The unique identifier for the patient.
    - analysis_data (dict): A dictionary containing the analysis results.
    
    Returns:
    - response (Response): The response object from the API request.
    """
    headers = {'Authorization': f'Bearer {API_KEY}'}
    payload = {
        'patient_id': patient_id,
        'analysis_data': analysis_data
    }
    
    # Make a POST request to the hospital's API to update the patient record
    response = requests.post(
        f"{HOSPITAL_API_BASE_URL}{UPDATE_RECORD_ENDPOINT}",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"Successfully updated record for patient {patient_id}.")
    else:
        print(f"Failed to update record for patient {patient_id}. Response code: {response.status_code}")
    
    return response

# Example usage
patient_id = '12345'
analysis_data = {
    'segmentation_results': 'path/to/results/file',
    'diagnosis': 'Findings from the image analysis...'
}

# Call the function to update the patient record with new analysis data
update_patient_record(patient_id, analysis_data)
