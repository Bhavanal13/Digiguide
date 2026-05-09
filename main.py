import os
import faiss
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
from PIL import Image
import io
from utils import image_to_embedding

# Global variables for index and metadata
index = None
metadata = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "vector_store.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

def load_index_and_metadata():
    global index, metadata
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print(f"Loading index from {INDEX_FILE}...")
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r') as f:
                temp_metadata = json.load(f)
                metadata = {int(k): v for k, v in temp_metadata.items()}
            print("Index and metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {e}")
    else:
        print("Index or metadata file not found. Please run build_index.py.")


# Knowledge Base for Landmark Details
LANDMARK_INFO = {
    "darmstadium": {
        "name": "Darmstadtium",
        "short_description": "Science & Congress Center",
        "long_description": "The Darmstadtium is a science and congress center in Darmstadt, Germany. Its name is derived from the chemical element darmstadtium, which was discovered in the city.",
        "history": "Opened in 2007, the Darmstadtium was built to serve as a hub for science and conferences in the 'City of Science'. It stands on the site of the former medieval city walls, parts of which are integrated into the modern architecture. The building was named after the chemical element Darmstadtium (Ds, atomic number 110), discovered at the nearby GSI Helmholtz Centre for Heavy Ion Research in 1994.",
        "facts": [
            "Named after the element Darmstadtium (110).",
            "Integrates genuine medieval city walls.",
            "Certified for sustainability.",
            "Total area of 18,000 square meters."
        ],
        "speech_text": "Welcome to the Darmstadtium. This modern science and congress center opened in 2007 and is named after the chemical element Darmstadtium, which was discovered right here in this city. Notice how parts of the medieval city walls are integrated into its modern glass architecture, blending history with innovation.",
        "lat": 49.8732,
        "lon": 8.6558
    },
    "mathildenhöhe": {
        "name": "Mathildenhöhe",
        "short_description": "Art Nouveau artists' colony",
        "long_description": "Mathildenhöhe represents a major center of Art Nouveau (Jugendstil) and was recognized as a UNESCO World Heritage Site in 2021.",
        "history": "Founded in 1899 by Grand Duke Ernst Ludwig, Mathildenhöhe was established as an artists' colony to promote modern art and design. Seven exhibitions were held between 1901 and 1914, leaving behind a unique architectural ensemble. It became a UNESCO World Heritage site in 2021, recognized as a pioneer of modernism.",
        "facts": [
            "UNESCO World Heritage Site since 2021.",
            "Home to the iconic Wedding Tower (Hochzeitsturm).",
            "Features the Russian Chapel with golden domes.",
            "Center of the Jugendstil (Art Nouveau) movement."
        ],
        "speech_text": "You are at Mathildenhöhe, a UNESCO World Heritage site and a masterpiece of Art Nouveau. Founded in 1899 as an artists' colony by Grand Duke Ernst Ludwig, this hill is famous for its Wedding Tower, looking like a hand with five fingers, and the golden-domed Russian Chapel. It is a stunning example of early modern architecture.",
        "lat": 49.8775,
        "lon": 8.6675
    },
    "orangerie_park": {
        "name": "Orangerie Park",
        "short_description": "Baroque park in Bessungen",
        "long_description": "The Orangerie in Darmstadt is a baroque building designed by architect Louis Remy de la Fosse. It is surrounded by a beautiful park.",
        "history": "Built between 1719 and 1721 by architect Louis Remy de la Fosse for Landgrave Ernst Ludwig. Originally designed to house exotic citrus plants during winter, it later served as a summer residence. The surrounding park was laid out in the French Baroque style and remains a popular recreational area for locals.",
        "facts": [
            "Designed by Louis Remy de la Fosse.",
            "Originally a shelter for orange trees.",
            "Located in the Bessungen district.",
            "Popular for summer picnics and concerts."
        ],
        "speech_text": "Welcome to the Orangerie Park in Bessungen. This beautiful baroque park surrounds the Orangerie building, which was constructed over 300 years ago to house citrus trees during the winter. Today, it is a favorite spot for locals to enjoy summer picnics and concerts amidst the French Baroque garden design.",
        "lat": 49.8596,
        "lon": 8.6565
    },
    "schloss": {
        "name": "Residenzschloss Darmstadt",
        "short_description": "Former residence of Landgraves",
        "long_description": "The Residential Palace Darmstadt is located in the center of the city. It was the residence of the Landgraves and Grand Dukes of Hesse-Darmstadt.",
        "history": "The castle's origins date back to the 13th century as a moated castle. It was transformed into a Renaissance residence in the 16th century and later expanded with Baroque elements. It served as the seat of government for the Landgraves and Grand Dukes of Hesse-Darmstadt for centuries until 1918.",
        "facts": [
            "Served as a residence for over 400 years.",
            "Houses the Technical University of Darmstadt library.",
            "Combines Renaissance and Baroque architecture.",
            "Partially destroyed in WWII and rebuilt."
        ],
        "speech_text": "You are standing before the Residenzschloss, the former residence of the Landgraves and Grand Dukes of Hesse-Darmstadt. Its history spans over 400 years, evolving from a medieval moated castle to the Renaissance and Baroque complex you see today. It now houses the university library and remains the heart of the city.",
        "lat": 49.8728,
        "lon": 8.6552
    },
    "waldspirale": {
        "name": "Waldspirale",
        "short_description": "Hundertwasser residential complex",
        "long_description": "The Waldspirale is a residential building complex in Darmstadt, designed by Austrian artist Friedensreich Hundertwasser. It features a unique spiral forest roof.",
        "history": "Completed in 2000, the Waldspirale ('Forest Spiral') was the final architectural design by Friedensreich Hundertwasser before his death. The building rejects straight lines and right angles, featuring over 1,000 unique windows and a green roof that spirals up to 12 stories high, allowing residents to walk upon it.",
        "facts": [
            "Has 105 apartments and no two windows are alike.",
            "Roof is planted with grass and trees.",
            "Missing straight lines and sharp corners.",
            "Design promotes harmony with nature."
        ],
        "speech_text": "This consists of the Waldspirale, or Forest Spiral. Designed by the famous artist Friedensreich Hundertwasser, this building rejects straight lines. Notice the colorful facade, the golden onions, and the fact that no two of its one thousand windows are the same. A forest literally grows on its spiral roof, symbolizing harmony with nature.",
        "lat": 49.8863,
        "lon": 8.6554
    },
    "unknown": {
        "name": "Unknown Landmark",
        "short_description": "Could not recognize",
        "long_description": "The AI could not identify this landmark with high confidence. Please try again from a different angle.",
        "history": "No history available for this unrecognized location.",
        "facts": [],
        "lat": 0.0,
        "lon": 0.0
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model (via utils) and Index
    try:
        # Trigger model load with a valid dummy image
        dummy_img = Image.new('RGB', (224, 224), color='white')
        image_to_embedding(dummy_img) 
        print("Model loaded and warmed up.")
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

    load_index_and_metadata()
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

# Mount dataset folder
if os.path.exists("dataset"):
    app.mount("/images", StaticFiles(directory="dataset"), name="images")
else:
    print("Warning: 'dataset' folder not found. Images will not be served.")

@app.get("/")
def root():
    return {"message": "DigiGuide backend running successfully!"}

@app.get("/weather")
def get_weather(lat: float, lon: float):
    """Returns weather information and visit recommendations using Open-Meteo API (free, no key needed)."""
    import requests
    
    try:
        # Open-Meteo API - completely free, no API key needed
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=auto"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current = data.get("current", {})
        temp = int(current.get("temperature_2m", 0))
        humidity = int(current.get("relative_humidity_2m", 0))
        weather_code = current.get("weather_code", 0)
        
        # Map WMO weather codes to conditions and icons
        # https://open-meteo.com/en/docs
        weather_map = {
            0: ("Clear", "☀️"),
            1: ("Mainly Clear", "🌤️"),
            2: ("Partly Cloudy", "⛅"),
            3: ("Overcast", "☁️"),
            45: ("Foggy", "🌫️"),
            48: ("Foggy", "🌫️"),
            51: ("Light Drizzle", "🌧️"),
            53: ("Drizzle", "🌧️"),
            55: ("Dense Drizzle", "🌧️"),
            61: ("Light Rain", "🌧️"),
            63: ("Rain", "🌧️"),
            65: ("Heavy Rain", "🌧️"),
            71: ("Light Snow", "🌨️"),
            73: ("Snow", "🌨️"),
            75: ("Heavy Snow", "🌨️"),
            80: ("Rain Showers", "🌦️"),
            81: ("Rain Showers", "🌦️"),
            82: ("Heavy Showers", "🌧️"),
            95: ("Thunderstorm", "⛈️"),
        }
        
        condition, icon = weather_map.get(weather_code, ("Unknown", "🌡️"))
        
        # Generate visit recommendation based on weather
        if weather_code in [0, 1]:
            recommendation = "Perfect weather for outdoor sightseeing! Don't forget sunscreen."
            best_time = "Morning (9-11 AM) or Late Afternoon (4-6 PM)"
            rating = "Excellent"
        elif weather_code in [2]:
            recommendation = "Great conditions for exploring. Enjoy the pleasant weather!"
            best_time = "Any time during daylight hours"
            rating = "Very Good"
        elif weather_code in [3, 45, 48]:
            recommendation = "Good for visiting. Overcast skies provide natural shade."
            best_time = "Midday is fine - no harsh sun"
            rating = "Good"
        elif weather_code in [51, 53, 55, 61, 63, 80, 81]:
            recommendation = "Light rain expected. Bring an umbrella for outdoor sites."
            best_time = "Check for rain breaks or visit covered areas"
            rating = "Fair"
        elif weather_code in [65, 82, 95]:
            recommendation = "Heavy weather expected. Consider indoor attractions today."
            best_time = "Wait for better conditions or visit museums"
            rating = "Poor"
        elif weather_code in [71, 73, 75]:
            recommendation = "Snowy conditions. Bundle up if visiting outdoor landmarks!"
            best_time = "Midday when it's warmest"
            rating = "Moderate"
        else:
            recommendation = "Check local conditions before your visit."
            best_time = "Flexible"
            rating = "Moderate"
        
        return {
            "temperature": temp,
            "condition": condition,
            "icon": icon,
            "humidity": humidity,
            "recommendation": recommendation,
            "bestTime": best_time,
            "rating": rating
        }
        
    except Exception as e:
        print(f"Weather API error: {e}")
        # Fallback response
        return {
            "temperature": 15,
            "condition": "Unknown",
            "icon": "🌡️",
            "humidity": 50,
            "recommendation": "Weather data unavailable. Check local forecasts.",
            "bestTime": "Flexible",
            "rating": "Unknown"
        }

@app.get("/landmarks")
def get_landmarks():
    """Returns a list of all known landmarks for the search bar."""
    results = []
    for key, info in LANDMARK_INFO.items():
        if key == "unknown": continue
        
        # Find an image asset for this landmark from metadata
        img_asset = ""
        for m_id, m_data in metadata.items():
            if m_data['landmark_name'] == key:
                img_asset = f"{key}/{m_data['filename']}"
                break
        
        results.append({
            "id": key,
            "name": info['name'],
            "shortDescription": info['short_description'],
            "longDescription": info['long_description'],
            "history": info.get('history', 'No history available.'),
            "facts": info.get('facts', []),
            "speechText": info.get('speech_text', ''),
            "lat": info['lat'],
            "lng": info['lon'],
            "imageAsset": img_asset,
            "matchType": "list",
            "distance": 0.0,
            "score": 1.0
        })
    return {"landmarks": results}

@app.post("/nearby")
def get_nearby_landmarks(
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Returns sorted list of nearby landmarks for Explore tab."""
    from math import radians, cos, sin, asin, sqrt
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 
        return c * r

    results = []
    for key, info in LANDMARK_INFO.items():
        if key == "unknown": continue
        
        l_lat = info.get('lat')
        l_lon = info.get('lon')
        
        dist = 0.0
        if l_lat and l_lon:
            dist = haversine(longitude, latitude, l_lon, l_lat)
        
        # Find image asset from metadata
        img_asset = ""
        for m_id, m_data in metadata.items():
             if m_data['landmark_name'] == key:
                 img_asset = f"{key}/{m_data['filename']}"
                 break
        
        results.append({
            "id": key,
            "name": info['name'],
            "shortDescription": info['short_description'],
            "longDescription": info['long_description'],
            "history": info.get('history', 'No history available.'),
            "facts": info.get('facts', []),
            "speechText": info.get('speech_text', ''),
            "lat": l_lat,
            "lng": l_lon,
            "imageAsset": img_asset,
            "matchType": "location",
            "distance": round(dist, 2), # Distance in km
            "score": 1.0
        })
    
    # Sort by distance
    results.sort(key=lambda x: x['distance'])
    
    return {"landmarks": results}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None)
):
    if index is None or not metadata:
        raise HTTPException(status_code=500, detail="Server not ready: Index not loaded.")

    try:
        print(f"DEBUG: Received request with lat={latitude}, lon={longitude}")
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get embedding
        query_emb = image_to_embedding(image).astype('float32')
        
        # Search (k=1)
        k = 1
        distances, indices = index.search(query_emb, k)
        
        idx = int(indices[0][0])
        score = float(distances[0][0])
        
        print(f"DEBUG: Best visual match: index={idx}, score={score}")

        VISUAL_THRESHOLD = 0.75 # Adjusted to 0.75 as per user request
        LOCATION_THRESHOLD_KM = 0.150 # 150 meters
        
        # Helper function for distance
        from math import radians, cos, sin, asin, sqrt
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a)) 
            r = 6371 
            return c * r

        # ---------------------------------------------------------
        # PRIORITY 1: VISUAL MATCH
        # ---------------------------------------------------------
        if score >= VISUAL_THRESHOLD and idx in metadata:
            match = metadata[idx]
            key = match['landmark_name']
            info = LANDMARK_INFO.get(key, LANDMARK_INFO["unknown"])
            
            print(f"DEBUG: Visual match candidate: {key} with score {score}")

            # Calculate distance if we have user location
            dist = 0.0
            if latitude and longitude and info.get('lat') and info.get('lon'):
                dist = haversine(longitude, latitude, info['lon'], info['lat'])
            
            # Logic Check: If visual match is very strong (>0.9), trust it.
            # If it's weaker (0.8-0.9) AND we have location, verify it's somewhat close (<10km).
            # This prevents "Darmstadium" (if in index) appearing when user is in a different city visually.
            is_valid_visual = True
            # RELAXED FOR TESTING: Commenting out strict distance check so user can test from laptop
            # if latitude and longitude and dist > 10.0 and score < 0.90:
            #      print(f"DEBUG: Rejecting visual match {key} because distance {dist:.2f}km is too far for score {score}")
            #      is_valid_visual = False

            if is_valid_visual:
                print(f"DEBUG: Returning VISUAL match: {info['name']}")
            return {"predictions": [{
                "id": str(idx),
                "name": info['name'],
                "shortDescription": info['short_description'],
                "longDescription": info['long_description'],
                "history": info.get('history', 'No history available.'),
                "facts": info.get('facts', []),
                "speechText": info.get('speech_text', ''),
                "lat": info['lat'],
                "lng": info['lon'],
                "imageAsset": f"{key}/{match['filename']}",
                "score": score,
                "matchType": "visual",
                "distance": round(dist, 2)
            }]}

        # ---------------------------------------------------------
        # PRIORITY 2: STRICT LOCATION MATCH (Only if no visual match)
        # ---------------------------------------------------------
        if latitude is not None and longitude is not None:
            closest_landmark = None
            min_dist = float('inf')
            
            for key, info in LANDMARK_INFO.items():
                if key == "unknown": continue
                
                l_lat = info.get('lat')
                l_lon = info.get('lon')
                
                if l_lat and l_lon:
                    dist = haversine(longitude, latitude, l_lon, l_lat)
                    if dist < min_dist:
                        min_dist = dist
                        closest_landmark = (key, info)
            
            # If closest landmark is within threshold, return it as the single result
            if closest_landmark and min_dist <= LOCATION_THRESHOLD_KM:
                key, info = closest_landmark
                
                # Find valid image
                img_asset = ""
                for m_id, m_data in metadata.items():
                    if m_data['landmark_name'] == key:
                        img_asset = f"{key}/{m_data['filename']}"
                        break
                
                print(f"DEBUG: Returning LOCATION match: {info['name']} (Dist: {min_dist}km)")
                return {"predictions": [{
                    "id": key,
                    "name": info['name'],
                    "shortDescription": info['short_description'],
                    "longDescription": info['long_description'],
                    "history": info.get('history', 'No history available.'),
                    "facts": info.get('facts', []),
                    "speechText": info.get('speech_text', ''),
                    "lat": info['lat'],
                    "lng": info['lon'],
                    "imageAsset": img_asset,
                    "score": 1.0, 
                    "matchType": "location",
                    "distance": round(min_dist, 2)
                }]}

        # ---------------------------------------------------------
        # PRIORITY 3: FALLBACK LIST (Sorted by distance)
        # ---------------------------------------------------------
        candidates = []
        if latitude is not None and longitude is not None:
             SEARCH_RADIUS_KM = 50.0
             
             for key, info in LANDMARK_INFO.items():
                 if key == "unknown": continue
                 
                 l_lat = info.get('lat')
                 l_lon = info.get('lon')
                 
                 if l_lat and l_lon:
                     dist = haversine(longitude, latitude, l_lon, l_lat)
                     if dist <= SEARCH_RADIUS_KM:
                         # Find image asset
                         img_asset = ""
                         for m_id, m_data in metadata.items():
                             if m_data['landmark_name'] == key:
                                 img_asset = f"{key}/{m_data['filename']}"
                                 break
                                 
                         candidates.append({
                             "id": key,
                             "name": info['name'],
                             "shortDescription": info['short_description'],
                             "longDescription": info['long_description'],
                             "history": info.get('history', 'No history available.'),
                             "facts": info.get('facts', []),
                             "speechText": info.get('speech_text', ''),
                             "lat": l_lat,
                             "lng": l_lon,
                             "imageAsset": img_asset,
                             "score": 0.0, 
                             "matchType": "list",
                             "distance": round(dist, 2)
                         })
            
             # Sort by distance and take top 5
             candidates.sort(key=lambda x: x['distance'])
             candidates = candidates[:5]
             
        # If absolutely nothing found (no location, no visual), return unknown
        if not candidates:
            info = LANDMARK_INFO["unknown"]
            print(f"DEBUG: Returning UNKNOWN match")
            candidates.append({
                "id": "-1",
                "name": info['name'],
                "shortDescription": info['short_description'],
                "longDescription": info['long_description'],
                "history": info.get('history', 'No history available.'),
                "facts": info.get('facts', []),
                "lat": latitude if latitude else 0.0,
                "lng": longitude if longitude else 0.0,
                "imageAsset": "",
                "score": 0.0,
                "matchType": "unknown",
                "distance": 0.0
            })

        print(f"DEBUG: Returning LIST with {len(candidates)} items")
        return {"predictions": candidates}

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
