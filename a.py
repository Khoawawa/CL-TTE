import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import os
import osmnx as ox
import networkx as nx

import pickle
data_path = "../../data/hcm_data/"
with open(os.path.join(data_path,"nwk_hcm/hcm_edges_new_simplify.pkl"), 'rb') as f:
    edgeinfo = pickle.load(f)
with open(os.path.join(data_path,"nwk_hcm/hcm_nodes_new.pkl"), 'rb') as f:
    nodeinfo = pickle.load(f)
    
min_lat, min_lon, max_lat, max_lon = float('inf'), float('inf'), float('-inf'), float('-inf')

for _, (lon, lat, _) in nodeinfo.items():
    min_lat = min(min_lat, lat)
    max_lat = max(max_lat, lat)
    min_lon = min(min_lon, lon)
    max_lon = max(max_lon, lon)

print(f"Bounding box: ({min_lat}, {min_lon}), ({max_lat}, {max_lon})")

bbox = (min_lon, min_lat, max_lon, max_lat)

graph = ox.graph_from_bbox(bbox=bbox, network_type='drive')

nodes, edges = ox.graph_to_gdfs(graph) 


valid_edges = []
invalid_edges = []

for i, edge in edgeinfo.items():
    u = int(edge[2])
    v = int(edge[3])
    
    if graph.has_edge(u, v):
        valid_edges.append((i, edge))
    else:
        if graph.has_edge(v, u):
            invalid_edges.append((i, edge,  "Exists, but opposite direction"))
        else:
            invalid_edges.append((i, edge, "Completely missing from graph"))

print("\n--- Validation Results ---")
print(f"Total Edges Checked: {len(edgeinfo)}")
print(f"Valid Edges: {len(valid_edges)}")
print(f"Invalid/Missing Edges: {len(invalid_edges)}")

if invalid_edges:
    print("\nSample of invalid edges:")
    for inv_edge in invalid_edges[:5]:
        print(f"Edge {inv_edge[0]}: {inv_edge[1][2]} -> {inv_edge[1][3]}: {inv_edge[2]}")