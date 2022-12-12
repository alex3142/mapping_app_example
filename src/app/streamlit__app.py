import pickle
from pathlib import Path

import folium
import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium


def get_map():

	# Create a Map instance
	i_map = folium.Map(
		location=[51.507741, -0.127487],
		zoom_start=12,
		control_scale=True,
		crs='EPSG3857'
	)
	folium.TileLayer('stamentoner').add_to(i_map)
	return i_map


st.set_page_config(
	page_title="fever-data-challenge",
	page_icon=":world_map:️",
	layout="wide",
)

st.markdown("""
# Researcher Location Plot

This plot shows a heat map of the likely locations of the
researcher. Where the more red the colour the more likely the 
researcher is to be in that square.

In the top right hand corner of the map there is a menu with
the different overlays for the map. The options are:

- combined: which is the combined probability distribution overlay
- combined_50%_probability: showing the total area where there is a 50% probability
- combined_90%_probability: showing the total area where there is a 90% probability 

""")


data_dir = Path(__file__).parent.parent.joinpath('data')
river_data = data_dir.joinpath('river.pkl')
boe_data = data_dir.joinpath('b_o_e.pkl')
satellite_data = data_dir.joinpath('satellite.pkl')
combined_data = data_dir.joinpath('combined.pkl')
combined_data_subset_50 = data_dir.joinpath('combined_subset_50.pkl')
combined_data_subset_90 = data_dir.joinpath('combined_subset_90.pkl')

default_data_to_show = 'combined_90%_probability'

data_store = {
	# 'boe': boe_data,
	# 'river': river_data,
	# 'satellite': satellite_data,
	'combined': combined_data,
	'combined_50%_probability': combined_data_subset_50,
	'combined_90%_probability': combined_data_subset_90,
}

data_gdfs = {
}
for data_name, file_path in data_store.items():
	with open(file_path, 'rb') as f:
		data_gdfs[data_name] = pickle.load(f)[['poly_earth', 'hex']].to_json()

i_map = get_map()

feature_groups = {}

for data_name, data_gdf in data_gdfs.items():
	feature_groups[data_name] = folium.FeatureGroup(
		name=data_name, show=data_name == default_data_to_show
	)
	feature_groups[data_name].add_child(
		folium.GeoJson(
			data_gdf,
			style_function=lambda feature: {
				'fillColor': feature['properties']['hex'],
				'color': feature['properties']['hex'],
				'weight': 0.5,
				'fillOpacity': 0.5,
			},
			name=data_name
		)
	)
	i_map.add_child(feature_groups[data_name])

for feat_group in feature_groups.values():
	i_map.add_child(feat_group)
folium.LayerControl().add_to(i_map)

# call to render Folium map in Streamlit
st_data = st_folium(i_map, width=700)
