import os

from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json


def init_environment(buildings_to_use, simulation_start_end=None, **kwargs):
    r"""Initialize `CityLearnEnv` and returns the environment

        Parameters
        ----------
        buildings_to_use: list[int]
            List to define which buildings are used in the environment, example: [1,2,4,17].
        simulation_start_end: list[int]
            List to define start and end time step, example: [0,8759].

        """
    schema_path = './data/citylearn_challenge_2022_phase_all/schema.json'
    schema = read_json(schema_path)
    if simulation_start_end is not None:
        schema['simulation_start_time_step'] = simulation_start_end[0]
        schema['simulation_end_time_step'] = simulation_start_end[1]
    dict_buildings = schema['buildings']

    # set all buildings to include=false
    for building_schema in dict_buildings.values():
        building_schema['include'] = False

    # include=true for buildings to use
    for building_number in buildings_to_use:
        building_id = 'Building_' + str(building_number)
        if building_id in dict_buildings:
            dict_buildings[building_id]['include'] = True

    env = CityLearnEnv(schema)
    return env


def get_string_file_size(file):
    file_size = os.stat(file).st_size
    if file_size > 1e+9:
        string_byte = "(" + str(round(file_size / 1e+9)) + " GB)"
    elif file_size > 1e+6:
        string_byte = "(" + str(round(file_size / 1e+6)) + " MB)"
    elif file_size > 1e+3:
        string_byte = "(" + str(round(file_size / 1e+3)) + " kB)"
    else:
        string_byte = "(" + str(round(file_size)) + " Byte)"

    return string_byte
