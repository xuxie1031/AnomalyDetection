import os
import configparser
import getopt
import sys

import numpy as np


def load_vessels_goals_dict(config_file):
    cparser = configparser.ConfigParser()
    cparser.read(config_file)

    base_path = cparser['GOAL_FILES']['BASE_PATH']
    goal_type = cparser['GOAL_FILES']['GOAL_TYPE']

    coords_dict_name = '{0}{1}'.format(base_path, '{0}{1}'.format('coords', goal_type))
    attract_flags_dict_name = '{0}{1}'.format(base_path, '{0}{1}'.format('attract_flags', goal_type))

    if not os.path.exists(coords_dict_name) or not os.path.exists(attract_flags_dict_name):
        return None, None

    goal_name_coords_dict = {}
    with open(coords_dict_name, 'r') as fid:
        for line in fid.readlines():
            name = line.split(' ')[0]
            coord = line.split(' ')[1].split(',')
            coord = np.array(coord).astype(np.float)
            goal_name_coords_dict[name] = coord
    fid.close()

    goal_name_attract_flags_dict = {}
    with open(attract_flags_dict_name, 'r') as fid:
        for line in fid.readlines():
            name = line.split(' ')[0]
            attract_flag = float(line.split(' ')[1])
            goal_name_attract_flags_dict[name] = attract_flag
    fid.close()

    print(goal_name_coords_dict)
    print(goal_name_attract_flags_dict)

    return goal_name_coords_dict, goal_name_attract_flags_dict


if __name__ == '__main__':
    config_file = None
    correct_str = '{0} -c <config file>'.format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:v", ["help", "config="])
    except getopt.GetoptError:
        print(correct_str)
        sys.exit(2)
    if len(opts) < 1:
        print(correct_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(correct_str)
            sys.exit()
        elif opt in ('-c', '--config'):
            config_file = arg
    
    if config_file is not None:
        load_vessels_goals_dict(config_file)