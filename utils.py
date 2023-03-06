from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json


def init_environment(buildings_to_use):
    r"""Initialize `CityLearnEnv` and returns the environment

        Parameters
        ----------
        buildings_to_use: list[int]
            List to define which buildings are used in the environment, example: [1,2,4,17].

        """
    schema_path = './data/citylearn_challenge_2022_phase_all/schema.json'
    schema = read_json(schema_path)
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
