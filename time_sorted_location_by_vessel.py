import os
import shutil
import re
import configparser
import sys
import getopt

import numpy as np


def point_in_box(bp1, bp2, test_pt):
    min_x, max_x = min(bp1[0], bp2[0]), max(bp1[0], bp2[0])
    min_y, max_y = min(bp1[1], bp2[1]), max(bp1[1], bp2[1])

    if min_x <= test_pt[0] <= max_x and min_y <= test_pt[1] <= max_y:
        return True
    return False


def split_file_by_vessel_with_loc(config_file):
    cparser = configparser.ConfigParser()
    cparser.read(config_file)

    base_path = cparser['SPLIT_FILES']['BASE_PATH']
    type_split = cparser['SPLIT_FILES']['TYPE_SPLIT']
    split_dir = '{0}{1}{2}{3}'.format(base_path, cparser['SPLIT_FILES']['DATA_FILE_NAME'][:-4], '_split_files', type_split)

    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    with open('{0}{1}'.format(base_path, cparser['SPLIT_FILES']['DATA_FILE_NAME'])) as fid:
        vessel_id_data_dict = {}

        bound1 = (float(cparser['SPLIT_FILES']['BOUND_LON1']), float(cparser['SPLIT_FILES']['BOUND_LAT1']))
        bound2 = (float(cparser['SPLIT_FILES']['BOUND_LON2']), float(cparser['SPLIT_FILES']['BOUND_LAT2']))

        re_str = '([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})'

        hdr = fid.readline()
        line_count = 1
        for line in fid.readlines():
            print('processing line id {0} ...'.format(line_count))
            line_count += 1
            data = line.split(',')
            mmsi = data[0]

            lat = float(data[2])
            lon = float(data[3])

            if point_in_box(bound1, bound2, (lon, lat)):

                time_stamp = data[1]
                split_vals = re.search(re_str, time_stamp).groups()
                year, month, day, hour, minute, second = split_vals
                day, hour, minute = int(day), int(hour), int(minute)

                sog = float(data[4])
                if sog < 1.0:
                    continue

                minute_val = hour*60+minute
                if mmsi not in vessel_id_data_dict.keys():
                    vessel_id_data_dict[mmsi] = []
                vessel_id_data_dict[mmsi].append([day, minute_val, lat, lon])
        
        for mmsi in vessel_id_data_dict.keys():
            vessel_file_name = '{0}/{1}'.format(split_dir, mmsi)
            data_arr = np.asarray(sorted(vessel_id_data_dict[mmsi]))
            np.savetxt(vessel_file_name, data_arr, delimiter=',')

if __name__ == '__main__':
    config_file = None
    correct_str = '{0} -c <config file>'.format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:", ["help", "config="])
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
        split_file_by_vessel_with_loc(config_file)