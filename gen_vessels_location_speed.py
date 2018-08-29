import os
import configparser
import sys
import getopt

import numpy as np


# note vessel data must contain at least two rows
def calc_vessel_coord_time_line(minute, vessel_minutes, vessel_data, minute_span_max):
    assert len(vessel_minutes) >= 2
    assert len(vessel_data) >= 2

    idx_lower = np.argwhere(vessel_minutes < minute).flatten()
    idx_upper = np.argwhere(vessel_minutes > minute).flatten()

    if idx_lower.shape[0] == 0:
        if vessel_minutes[1]-vessel_minutes[0] > minute_span_max or vessel_minutes[0]-minute > minute_span_max:
            return None, None

        vessel_speed = (vessel_data[1, 2:]-vessel_data[0, 2:])/(vessel_minutes[1]-vessel_minutes[0])
        vessel_coord = vessel_data[0, 2:]-vessel_speed*(vessel_minutes[0]-minute)

    elif idx_upper.shape[0] == 0:
        if vessel_minutes[-1]-vessel_minutes[-2] > minute_span_max or minute-vessel_minutes[-1] > minute_span_max:
            return None, None

        vessel_speed = (vessel_data[-1, 2:]-vessel_data[-2, 2:])/(vessel_minutes[-1]-vessel_minutes[-2])
        vessel_coord = vessel_data[-1, 2:]+vessel_speed*(minute-vessel_minutes[-1])

    else:
        idx_lower_bound = idx_lower[-1]
        idx_upper_bound = idx_upper[0]
        if vessel_minutes[idx_upper_bound]-vessel_minutes[idx_lower_bound] > minute_span_max:
            return None, None

        vessel_speed = (vessel_data[idx_upper_bound, 2:]-vessel_data[idx_lower_bound, 2:])/(vessel_minutes[idx_upper_bound]-vessel_minutes[idx_lower_bound])
        vessel_coord = vessel_data[idx_lower_bound, 2:]+vessel_speed*(minute-vessel_minutes[idx_lower_bound])

    return vessel_coord, vessel_speed


def gen_coords_time_window(config_file, intention_window_flag):
    cparser = configparser.ConfigParser()
    cparser.read(config_file)

    base_path = cparser['SPLIT_FILES']['BASE_PATH']
    type_split = cparser['SPLIT_FILES']['TYPE_SPLIT']
    split_dir = '{0}{1}{2}{3}'.format(base_path, cparser['SPLIT_FILES']['DATA_FILE_NAME'][:-4], '_split_files', type_split)

    day_min = int(cparser[intention_window_flag]['DAY_MIN'])
    day_max = int(cparser[intention_window_flag]['DAY_MAX'])
    minute_min = int(cparser[intention_window_flag]['MINUTE_MIN'])
    minute_max = int(cparser[intention_window_flag]['MINUTE_MAX'])
    minute_step = int(cparser[intention_window_flag]['MINUTE_STEP'])
    minute_span_max = int(cparser[intention_window_flag]['MINUTE_SPAN_MAX'])

    min_scale = 24*60
    if ((day_max-1)*min_scale+minute_max) - ((day_min-1)*min_scale+minute_min) > minute_span_max:
        return None, None
    minutes = range((day_min-1)*min_scale+minute_min, (day_max-1)*min_scale+minute_max, minute_step)

    vessels_coords = []
    vessels_speeds = []
    vessels_accepts = []
    accept_flag = True
    for name in os.listdir(split_dir):
        filename = '{0}/{1}'.format(split_dir, name)
        print('flag {0} extracting {1}...'.format(intention_window_flag, filename))
        vessel_data = np.genfromtxt(filename, delimiter=',')
        if len(vessel_data.shape) == 1 or len(vessel_data) < 2:
            continue

        vessel_minutes = (vessel_data[:, 0]-1)*min_scale+vessel_data[:, 1]

        vessel_coords = []
        vessel_speeds = []
        for minute in minutes:
            vessel_coord, vessel_speed = calc_vessel_coord_time_line(minute, vessel_minutes, vessel_data, minute_span_max)
            if vessel_coord is None:
                accept_flag = False
                break
            vessel_coords.append(vessel_coord)
            vessel_speeds.append(vessel_speed)

        if accept_flag == True:
            vessel_coords, vessel_speeds = np.asarray(vessel_coords), np.asarray(vessel_speeds)
            vessels_coords.append(vessel_coords)
            vessels_speeds.append(vessel_speeds)
            vessels_accepts.append(name)
        accept_flag = True


    return np.asarray(vessels_coords), np.asarray(vessels_speeds), vessels_accepts
    

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
        vessels_coords, _, _ = gen_coords_time_window(config_file, 'TRAINING_INTENTION_WINDOW')
        vessels_coords_test, _, _ = gen_coords_time_window(config_file, 'TESTING_INTENTION_WINDOW')

        print('flag training vessels_coords count {0}'.format(len(vessels_coords)))
        print('flag testing vessels_coords count {0}'.format(len(vessels_coords_test)))