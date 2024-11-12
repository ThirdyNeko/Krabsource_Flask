from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
from io import BytesIO
from fastai.vision.all import *
from collections import defaultdict
import pandas as pd
import folium
import os
from geopy.distance import geodesic

import numpy as np
import cv2

#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

# Initialize Flask app
app = Flask(__name__)

# Load your ResNet-18 model
model_path = os.path.join(os.path.dirname(__file__), 'krabsource.pkl')
learn = load_learner(model_path)

# Dictionary to store crab counts by GPS location and type
crab_counts = defaultdict(lambda: defaultdict(int))

# Define color ranges for classification
color_ranges = {
    'Blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
    'Green': (np.array([30, 50, 50]), np.array([80, 255, 255])),
    'Brown': (np.array([0, 50, 50]), np.array([20, 255, 150])),
    'Red': (np.array([0, 50, 50]), np.array([20, 255, 255])),
    'Yellow': (np.array([20, 50, 50]), np.array([40, 255, 255])),
    'Orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
}

def preprocess_image(image_file):
    img = Image.open(BytesIO(image_file.read()))
    img = img.resize((224, 224))  # Resize to match model input size
    return img

def detect_colors_and_shapes(image):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_areas = {}
    shape_classifications = {}

    for color_name, (lower, upper) in color_ranges.items():
        color_mask = cv2.inRange(hsv_image, lower, upper)
        color_areas[color_name] = np.sum(color_mask)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            num_sides = len(approx)
            if num_sides >= 7:
                shape_classifications[color_name] = 'Rounded Lobes'
            elif num_sides <= 6:
                shape_classifications[color_name] = 'Sharp Edges'
            else:
                shape_classifications[color_name] = 'Oval'

    dominant_color = max(color_areas, key=color_areas.get, default="Unknown")
    shape_classification = shape_classifications.get(dominant_color, "Not classified")
    return dominant_color, shape_classification

@app.route('/classify', methods=['POST'])
def classify_image():
    image_file = request.files['image']
    latitude = request.form['latitude']
    longitude = request.form['longitude']

    img = preprocess_image(image_file)
    pred_class, pred_idx, outputs = learn.predict(img)
    probability = float(outputs[pred_idx]) * 100

    if probability < 60:
        return jsonify({
            '': 'Take a better image',
            'Confidence': probability,
        })

    crab_counts[(latitude, longitude)][str(pred_class)] += 1
    dominant_color, shape_classification = detect_colors_and_shapes(img)

    response_data = {
        'Species': str(pred_class),
        'Confidence': probability,
        'Color': dominant_color,
        'Shape': shape_classification,
    }

    return jsonify(response_data)

# Endpoint to fetch crab data for map display
@app.route('/get_crab_data', methods=['GET'])
def get_crab_data():
    data = []
    for (latitude, longitude), species_counts in crab_counts.items():
        for species, count in species_counts.items():
            data.append({
                'latitude': float(latitude),
                'longitude': float(longitude),
                'species': species,
                'count': count
            })
    return jsonify(data)

# Function to generate and save crab counts to Excel file
def save_crab_counts_to_excel(distance_threshold=0.01):  # Threshold in degrees (approximately 1 km)
    data = []
    visited_locations = {}

    for (latitude, longitude), species_counts in crab_counts.items():
        latitude = float(latitude)
        longitude = float(longitude)
        current_location = (latitude, longitude)
        found_nearby = False

        # Check if the current location is near any of the visited locations
        for visited_loc, species_data in visited_locations.items():
            if geodesic(current_location, visited_loc).meters < distance_threshold * 111320:
                found_nearby = True
                # Update counts for each species at the nearby location
                for species, count in species_counts.items():
                    if species in species_data:
                        species_data[species] += count  # Add to existing count for this species
                    else:
                        species_data[species] = count  # Add new species with its count
                break
        
        if not found_nearby:
            # If no nearby location found, add this location as a new entry in visited_locations
            visited_locations[current_location] = species_counts.copy()

    # Convert collected data to a DataFrame
    for (latitude, longitude), species_data in visited_locations.items():
        for species, count in species_data.items():
            data.append([latitude, longitude, species, count])

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Species', 'Count'])
    excel_path = os.path.abspath('crab_counts.xlsx')
    df.to_excel(excel_path, index=False)

    print(f"Data saved to {excel_path}")


@app.route('/get_excel', methods=['GET'])
def get_excel():
    save_crab_counts_to_excel()
    excel_path = os.path.abspath('crab_counts.xlsx')
    return send_file(excel_path, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/species_map', methods=['GET']) 
def species_map():
    # Load data from Excel file
    excel_file = 'crab_counts.xlsx'  # Replace with the path to your Excel file
    df = pd.read_excel(excel_file)

    # Sample structure: Assuming the Excel has columns 'Species', 'Latitude', 'Longitude', 'Count'
    species_data = df.to_dict(orient='records')  # Convert to a list of dictionaries

    # Initialize the map centered on a region
    mymap = folium.Map(location=[10.7393933, 122.5675356], zoom_start=15, tiles='OpenStreetMap')

    # Define colors for each species
    species_colors = {
        "Kasag (female)": "pink",
        "Kasag (male)": "blue",
        "Alimango": "red",
        "Dawat (adult)": "green",
        "Dawat (juvenile)": "lightgreen",
        "Kumong": "purple",
        "Kurusan": "orange",
        "Kalintugas": "yellow"
    }

    # Add bubbles for each location from the Excel data
    for entry in species_data:
        species = entry["Species"]
        color = species_colors.get(species, "black")  # Default to black if species is not in colors

        # Debugging: Print each marker's data to confirm it's working
        print(f"Adding marker for {species} at ({entry['Latitude']}, {entry['Longitude']}), count: {entry['Count']}")

        folium.CircleMarker(
            location=(entry["Latitude"], entry["Longitude"]),
            radius=entry["Count"],  # Bubble size based on count
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{species} ({entry['Count']} individuals)",
            fill_opacity=0.7,
            weight=0
        ).add_to(mymap)

    # Get HTML representation of the map
    map_html = mymap._repr_html_()

    # Return HTML directly to render map in Flask
    return map_html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
