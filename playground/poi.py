import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import osmnx as ox
from shapely.geometry import Point, LineString
import os

# Define the path to your data
print(os.getcwd())
data_path = "../data/hcm_data" # hcm data
# data_path = ""
if 'hcm' in data_path: 
    node_path = "nwk_hcm/hcm_nodes.shp"
    edge_path = "nwk_hcm/hcm_edges.shp"
    ox.settings.overpass_settings = f'[out:json][timeout:60][date:"2026-0-10T15:00:00Z"]'

else:
    node_path = "porto_network/porto_nodes.shp"
    edge_path = "porto_network/porto_edges.shp"
    
# Load the node and edge data
our_nodes = gpd.read_file(os.path.join(data_path, node_path))
our_edges = gpd.read_file(os.path.join(data_path, edge_path))
#extract the bounding box of our nodes
min_lat = our_nodes.y.min()
max_lat = our_nodes.y.max()
min_lon = our_nodes.x.min()
max_lon = our_nodes.x.max()
pad = 0.01  # Add a small padding to ensure we capture all nodes    
bbox = (min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad)
# define the tags we want to extract from OSM
tags = {
    "amenity": [
        # catering
        "restaurant", "cafe", "fast_food", "pub", "bar", "biergarten",

        # education
        "school", "college", "kindergarten", "university",

        # medical
        "hospital",

        # transport
        "bus_station", "taxi",

        # parking
        "parking",

        # retail centers
        "shopping_centre",
    ],
    
    "shop": [
        "supermarket", "convenience",
        "hairdresser", "laundry", "beauty",
    ],

    "highway": [
        "bus_stop",
    ],

    "office": True,

    "landuse": [
        "industrial",
    ]
}
# Extract POIs for each tag and store them in a dictionary
gdf_tags = {}

for tag, value in tags.items():
    try:
        gdf_tag = ox.features_from_bbox(bbox=bbox, tags={tag: value})
        gdf_tags[tag] = gdf_tag
    except Exception as e:
        print(f"{tag} failed:", e)
        
gdf_pois = pd.concat(gdf_tags.values())
gdf_pois = gdf_pois.reset_index()

gdf_pois_reduced = gdf_pois[gdf_pois.geometry.notnull()]
gdf_pois_reduced = gdf_pois_reduced.set_crs(epsg=4326, allow_override=True)

cols = ["element","id","geometry","amenity","shop","highway","office","landuse"]
gdf_pois_reduced = gdf_pois_reduced[[c for c in cols if c in gdf_pois_reduced.columns]]

gdf_pois_reduced['poi_type'] = (
    gdf_pois_reduced['amenity']
    .fillna(gdf_pois_reduced['shop'])
    .fillna(gdf_pois_reduced['highway'])
    .fillna(gdf_pois_reduced['office'])
    .fillna(gdf_pois_reduced['landuse'])
)

gdf_pois_reduced.drop(columns=["amenity","shop","highway","office","landuse"], inplace=True)
