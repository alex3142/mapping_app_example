from typing import (
    Tuple,
    Union,
    Dict,
    Any,
)
from pathlib import Path
import sys
import os
import logging

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.stats import lognorm, norm
import yaml

# need to add src directory to path to import locally if
# not install or source root defined
src_folder_path = str(Path(os.getcwd()).parent.parent)
if src_folder_path not in sys.path:
    sys.path.append(src_folder_path)

from fever_challenge.fever_utils.utils import (
    EuclideanEarthTransformer,
    save_pickle,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_AND_CONST_FILEPATH = Path(os.getcwd()).parent.joinpath('data_and_constants')
COORDS_DATA_PATH = DATA_AND_CONST_FILEPATH.joinpath('lat_long_coordinates_researcher_location.csv')
CONSTANTS_LOCATION = DATA_AND_CONST_FILEPATH.joinpath('constants.yaml')

GDF_SAVE_FILEPATH = Path(os.getcwd()).parent.parent.joinpath('data')

SAVE_LOCATIONS = {
    'satellite': GDF_SAVE_FILEPATH.joinpath('satellite.pkl'),
    'b_o_e': GDF_SAVE_FILEPATH.joinpath('b_o_e.pkl'),
    'river': GDF_SAVE_FILEPATH.joinpath('river.pkl'),
    'combined': GDF_SAVE_FILEPATH.joinpath('combined.pkl'),
    'combined_subset_50': GDF_SAVE_FILEPATH.joinpath('combined_subset_50.pkl'),
    'combined_subset_90': GDF_SAVE_FILEPATH.joinpath('combined_subset_90.pkl'),
}


def get_constants(const_file_path: Union[Path, str]) -> Dict[str, Any]:
    """
    This function parses the constants file to a dictionary
    :param const_file_path: Union[Path, str] - file path to const data
    :return:
    """
    with open(const_file_path, 'r') as file:
        constants_data = yaml.safe_load(file)
    return constants_data


def get_coords(coords_file_path: Union[Path, str]) -> Dict[str, Any]:
    """
    This function gets the required coordinate data from the filepath
    :param coords_file_path: Union[Path, str] - file path to coords data
    :return:
    """
    coords_data = pd.read_csv(coords_file_path)
    coords_data_b_o_e = coords_data.query('description == "bank_of_england"')
    coords_data_b_o_e = tuple(coords_data_b_o_e[['lat', 'long']].values[0])
    coords_data_satellite = coords_data.query('description == "satellite_path"')
    coords_data_river = coords_data.query('description == "river_thames"')

    return {
        'b_o_e': coords_data_b_o_e,
        'satellite': coords_data_satellite,
        'river': coords_data_river
    }


def subset_data(input_gdf: gpd.GeoDataFrame, threshold: float, col_name_to_limit: str) -> gpd.GeoDataFrame:
    input_gdf_copy = input_gdf.copy()

    input_gdf_copy = input_gdf_copy.sort_values(by=[col_name_to_limit], ascending=False)
    input_gdf_copy['cumsum'] = input_gdf_copy[col_name_to_limit].cumsum()
    return input_gdf_copy.query('cumsum <= @threshold')


def colour_grid(
        grid_input: gpd.GeoDataFrame,
        col_name_to_colour: str,
        col_name_output: str = 'hex',
        colourmap: str = 'rainbow'
) -> gpd.GeoDataFrame:
    """
    This function colours the grid proportional to he normalised given column
    :param grid_input:
    :param col_name_to_colour:
    :param col_name_output:
    :param colourmap:
    :return:
    """
    grid = grid_input.copy()
    norm = mpl.colors.Normalize(vmin=grid[col_name_to_colour].min(), vmax=grid[col_name_to_colour].max())
    cmap = getattr(cm, colourmap)

    col_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    grid[col_name_output] = grid[col_name_to_colour].apply(lambda x: mpl.colors.to_hex(col_mapper.to_rgba(x)))
    return grid


def get_b_o_e_distribution(
        base_gdf: gpd.GeoDataFrame,
        coords_data_b_o_e: Tuple[float, float],
        euc_trans: EuclideanEarthTransformer,
        model_constants_data: Dict[str, any],
) -> gpd.GeoDataFrame:
    """
    This function gets the distribution for the bank of england distribution
    :param base_gdf:
    :param coords_data_b_o_e:
    :param euc_trans:
    :param model_constants_data:
    :return:
    """
    b_o_e_gdf = base_gdf.copy()

    point_euc_coords_data_b_o_e = Point(euc_trans.earth_to_euclidean(lat_long=coords_data_b_o_e))

    b_o_e_gdf['distance_from_boe'] = b_o_e_gdf['geometry'].distance(point_euc_coords_data_b_o_e)
    b_o_e_gdf['b_o_e_distribution'] = lognorm.pdf(
        b_o_e_gdf['distance_from_boe'],
        scale=np.exp(model_constants_data['b_o_e_data']['mean']),
        s=model_constants_data['b_o_e_data']['stdev']
    )
    b_o_e_gdf = colour_grid(b_o_e_gdf, 'b_o_e_distribution')
    return b_o_e_gdf


def get_satellite_distribution(
        base_gdf: gpd.GeoDataFrame,
        coords_data_satellite: pd.DataFrame,
        euc_trans: EuclideanEarthTransformer,
        model_constants_data: Dict[str, any],
) -> gpd.GeoDataFrame:

    coords_data_satellite['x_y'] = coords_data_satellite.apply(
        lambda row: euc_trans.earth_to_euclidean(lat_long=(row['lat'], row['long'])), axis=1)
    satelite_path_line = LineString(list(coords_data_satellite['x_y'].values))

    satellite_gdf = base_gdf.copy()
    satellite_gdf['distance_from_satellite'] = satellite_gdf['geometry'].distance(satelite_path_line)

    satellite_gdf['satellite_distribution'] = satellite_gdf['distance_from_satellite'].apply(
        lambda x: norm.pdf(x, scale=model_constants_data['satellite_data']['stdev'])
    )
    satellite_gdf = colour_grid(satellite_gdf, 'satellite_distribution')

    return satellite_gdf


def get_river_distribution(
        base_gdf: gpd.GeoDataFrame,
        coords_data_river: pd.DataFrame,
        euc_trans: EuclideanEarthTransformer,
        model_constants_data: Dict[str, any],
) -> gpd.GeoDataFrame:

    coords_data_river['x_y'] = coords_data_river.apply(
        lambda row: euc_trans.earth_to_euclidean(lat_long=(row['lat'], row['long'])), axis=1)
    river_path_line = LineString(list(coords_data_river['x_y'].values))

    river_gdf = base_gdf.copy()
    river_gdf['distance_from_river'] = river_gdf['geometry'].distance(river_path_line)
    river_gdf['river_distribution'] = river_gdf['distance_from_river'].apply(
        lambda x: norm.pdf(x, scale=model_constants_data['river_data']['stdev']))
    river_gdf = colour_grid(river_gdf, 'river_distribution')

    return river_gdf


def combined_distributions(
    base_gdf: gpd.GeoDataFrame,
    satellite_gdf: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    b_o_e_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    temp_sat = satellite_gdf['satellite_distribution'] * (
            1 / satellite_gdf['satellite_distribution'].sum()
    )
    temp_b_o_e = b_o_e_gdf['b_o_e_distribution'] * (1 / b_o_e_gdf['b_o_e_distribution'].sum())
    temp_river = river_gdf['river_distribution'] * (1 / river_gdf['river_distribution'].sum())
    temp_combined = temp_sat * temp_b_o_e * temp_river
    temp_combined = temp_combined * (1 / temp_combined.sum())

    combined_gdf = base_gdf.copy()
    combined_gdf['combined_distribution'] = temp_combined
    combined_gdf = colour_grid(combined_gdf, 'combined_distribution')
    return combined_gdf


def main():

    model_constants_data = get_constants(const_file_path=CONSTANTS_LOCATION)
    coords_data = get_coords(coords_file_path=COORDS_DATA_PATH)

    euc_trans = EuclideanEarthTransformer(
        lat_min=model_constants_data['area_bounds']['lat_min'],
        lat_max=model_constants_data['area_bounds']['lat_max'],
        long_min=model_constants_data['area_bounds']['long_min'],
        long_max=model_constants_data['area_bounds']['long_max'],
        lat_scaler=model_constants_data['area_scales']['lat'],
        long_scaler=model_constants_data['area_scales']['long'],
    )

    data_gdf = euc_trans.get_euclidean_grid(n_squares_in_x=200)
    b_o_e_data_gdf = get_b_o_e_distribution(
        base_gdf=data_gdf,
        coords_data_b_o_e=coords_data['b_o_e'],
        euc_trans=euc_trans,
        model_constants_data=model_constants_data,
    )

    satellite_data_gdf = get_satellite_distribution(
        base_gdf=data_gdf,
        coords_data_satellite=coords_data['satellite'],
        euc_trans=euc_trans,
        model_constants_data=model_constants_data,
    )

    river_data_gdf = get_river_distribution(
        base_gdf=data_gdf,
        coords_data_river=coords_data['river'],
        euc_trans=euc_trans,
        model_constants_data=model_constants_data,
    )

    combined_gdf = combined_distributions(
        base_gdf=data_gdf,
        satellite_gdf=satellite_data_gdf,
        river_gdf=river_data_gdf,
        b_o_e_gdf=b_o_e_data_gdf,
    )

    combined_subsets_thresholds = {
        'combined_subset_50': 0.5,
        'combined_subset_90': 0.9,
    }

    combined_subsets = {}
    for subset_name, threshold in combined_subsets_thresholds.items():
        combined_subsets[subset_name] = subset_data(combined_gdf, threshold, 'combined_distribution')

    distributions_data = {
        'river': river_data_gdf,
        'satellite': satellite_data_gdf,
        'b_o_e': b_o_e_data_gdf,
        'combined': combined_gdf,
        'combined_subset_50': combined_subsets['combined_subset_50'],
        'combined_subset_90': combined_subsets['combined_subset_90']
    }

    for i_dist_name, i_dist_data in distributions_data.items():
        save_pickle(data=i_dist_data, filename=SAVE_LOCATIONS[i_dist_name])


if __name__ == '__main__':
    log.info('running...')
    main()
    log.info('done.')
