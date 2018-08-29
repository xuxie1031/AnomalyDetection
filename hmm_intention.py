import numpy as np
import threading
from sklearn.svm import OneClassSVM

from gen_vessels_location_speed import *
from load_vessels_goals import *
from posterior_psi import *

class HMMVesselIntention(threading.Thread):
    def __init__(self, vessels_coords, vessels_speeds, vessels_coords_test, vessels_speeds_test, goal_name_coords_dict, goal_name_attract_flags_dict, attract_sigma=1.0, disattract_sigma=1.0):
        threading.Thread.__init__(self)
        self.vessels_coords = vessels_coords
        self.vessels_speeds = vessels_speeds

        self.attract_sigma = attract_sigma
        self.disattract_sigma = disattract_sigma

        self.goal_coords = []
        for name in goal_name_coords_dict.keys():
            goal_name_coord = np.asarray(goal_name_coords_dict[name])
            self.goal_coords.append(goal_name_coord)
        self.goal_coords = np.asarray(self.goal_coords)

        self.attract_flags = []
        for name in goal_name_attract_flags_dict.keys():
            goal_name_attract_flag = goal_name_attract_flags_dict[name]
            self.attract_flags.append(goal_name_attract_flag)
        self.attract_flags = np.asarray(self.attract_flags)

        self.attract_sigmas = np.zeros(self.attract_flags.shape)
        self.attract_sigmas[self.attract_flags < 0] = disattract_sigma
        self.attract_sigmas[self.attract_flags > 0] = attract_sigma

        vessel_num = len(self.vessels_coords)
        self.vessel_omega_group_id = np.zeros(vessel_num)   # same group id for all vessels
        self.group_num = len(np.unique(self.vessel_omega_group_id))
        omega_dim = self.attract_flags.shape[0]+self.group_num
        self.omega = np.random.randn(omega_dim)
        self.attract_omega = self.omega[:-self.group_num]

        self.load_test_data(vessels_coords_test, vessels_speeds_test)


    def load_test_data(self, vessels_coords, vessels_speeds):
        self.vessels_coords_test = vessels_coords
        self.vessels_speeds_test = vessels_speeds
        
        vessel_num = len(self.vessels_coords_test)
        self.vessel_omega_group_id_test = np.zeros(vessel_num)  # same group id for all vessels


    def run(self):
        ''' Proposed Method '''

        # Training

        # while True:
        for i in range(10):
            print('training epoch {0}...'.format(i))

            # calculate features
            features_psi, _ = psi_with_fixed_horizon_greedy(self.goal_coords, self.vessels_coords, self.vessels_speeds, self.attract_flags, self.attract_sigmas, self.attract_omega, self.vessel_omega_group_id, self.disattract_sigma, L=self.vessels_coords.shape[1])

            # one-class SVM training
            clf = OneClassSVM(kernel='linear', nu=0.1)
            clf.fit(features_psi)
            self.omega = clf.coef_.flatten()
            self.attract_omega = self.omega[:-self.group_num]


        # Testing

        anomal_pred = self.predict(clf)
        print(anomal_pred)


        # Baseline Method, one-class SVM on trajectory points only will be added


    def predict(self, clf):
        # calculate features
        # same set of time interval, with variant vessel num and vessel group assignment

        features_psi, _ = psi_with_fixed_horizon_greedy(self.goal_coords, self.vessels_coords_test, self.vessels_speeds_test, self.attract_flags, self.attract_sigmas, self.attract_omega, self.vessel_omega_group_id, self.disattract_sigma, L=self.vessels_coords_test.shape[1])

        # one-class SVM predicting
        anomal_pred = clf.predict(features_psi)

        return anomal_pred


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
        vessels_coords, vessels_speeds, _ = gen_coords_time_window(config_file, 'TRAINING_INTENTION_WINDOW')
        vessels_coords_test, vessels_speeds_test, _ = gen_coords_time_window(config_file, 'TESTING_INTENTION_WINDOW')
        goal_name_coords_dict,  goal_name_attract_flags_dict = load_vessels_goals_dict(config_file)

        vessel_intention = HMMVesselIntention(vessels_coords, vessels_speeds, vessels_coords_test, vessels_speeds_test, goal_name_coords_dict, goal_name_attract_flags_dict, attract_sigma=1e2, disattract_sigma=1e-1)
        vessel_intention.daemon = True
        vessel_intention.start()

        vessel_intention.join()
        print('anomaly prediction end...')