import requests
import argparse
import time

def test_api(image_path):
    url = "http://127.0.0.1:8000/predict"
    
    print(f"Sending request to {url}...")
    start_time = time.time()
    
    try:
        data = {}
        if args.lat is not None and args.lon is not None:
             data['latitude'] = args.lat
             data['longitude'] = args.lon
             print(f"Sending with Location: {args.lat}, {args.lon}")

        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, params=data) # FastAPI handles non-body params as query or form. 
                                                                    # Since we are using UploadFile, other params usually go to query params or form-data.
                                                                    # requests.post(..., data=data) sends form-data. 
                                                                    # FastAPI Endpoint: latitude: float = None ... -> Query param by default? 
            # Wait, in FastAPI:
            # def predict(file: UploadFile, latitude: float = Form(...)) -> Form data
            # def predict(file: UploadFile, latitude: float = Query(...)) -> Query param
            # In my implementation: latitude: float = None -> This is a Query parameter by default in FastAPI!
            # So passing as params=data is correct for Query parameters.
            
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            print(f"Success! Response time: {duration:.4f} seconds")
            print("Response:", response.json())
        else:
            print(f"Failed (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is uvicorn running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test API recognition")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--lat", type=float, help="Latitude", default=None)
    parser.add_argument("--lon", type=float, help="Longitude", default=None)
    args = parser.parse_args()
    
    test_api(args.image)
