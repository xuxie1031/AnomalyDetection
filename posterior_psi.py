import numpy as np

def goal_force_euclidean(goal_coords, vessel_coord, attract_flags, attract_sigmas, epsilon=1e-2):
    goal_dir = goal_coords-vessel_coord
    goal_dir_norm = np.maximum(epsilon, np.linalg.norm(goal_dir, axis=1))
    force_dirs = goal_dir / goal_dir_norm[:, None]
    force_magnitudes = (attract_sigmas/goal_dir_norm)**2 * attract_flags

    return force_magnitudes, force_dirs


def step_psi_with_selected_latent(force_magnitudes, force_dirs, vessel_speed, latent_flags):
    return latent_flags*force_magnitudes*np.sum(force_dirs*vessel_speed, axis=1)


def psi_with_fixed_horizon_greedy(goal_coords, vessels_coords, vessels_speeds, attract_flags, attract_sigmas, attract_omega, vessel_omega_group_id, disattract_sigma, L=50):
    vessel_num = vessels_coords.shape[0]
    group_num = len(np.unique(vessel_omega_group_id))

    attract_latent_flags = np.zeros(attract_flags.shape)
    vessel_latent_flags = np.ones(vessel_num)
    vessel_flags = -1.0*np.ones(vessel_num)
    vessel_sigmas = disattract_sigma*np.ones(vessel_num)

    attracts_psi = []
    attracts_latent_ids = []
    vessels_psi = np.zeros((vessel_num, group_num))

    for v in range(len(vessels_coords)):
        print('calc feature of vessel {0}...'.format(v))
        accu_attract_psi = np.zeros(attract_flags.shape)
        latent_ids = []
        vessel_psi_value = 0.0

        for i in range(min(L, vessels_coords.shape[1])):
            vessel_coord = vessels_coords[v, i, :]
            vessel_speed = vessels_speeds[v, i, :]

            attract_force_magnitudes, attract_force_dirs = goal_force_euclidean(goal_coords, vessel_coord, attract_flags, attract_sigmas)

            # for 
            max_step_log_posterior = -np.inf
            max_step_psi = np.zeros(attract_flags.shape)
            latent_id = -1
            for idx in range(len(attract_latent_flags)):
                new_latent_flags = np.copy(attract_latent_flags)
                new_latent_flags[idx] = 1.0
                tmp_psi = step_psi_with_selected_latent(attract_force_magnitudes, attract_force_dirs, vessel_speed, new_latent_flags)
                if attract_omega.dot(tmp_psi) > max_step_log_posterior:
                    max_step_log_posterior = attract_omega.dot(tmp_psi)
                    max_step_psi = tmp_psi
                    latent_id = idx

            accu_attract_psi += max_step_psi
            latent_ids.append(latent_id)
        
            vessel_force_magnitudes, vessel_force_dirs = goal_force_euclidean(vessels_coords[:, i, :], vessel_coord, vessel_flags, vessel_sigmas)
            vessel_psi = step_psi_with_selected_latent(vessel_force_magnitudes, vessel_force_dirs, vessel_speed, vessel_latent_flags)
            vessel_psi_value += np.sum(vessel_psi)

        attracts_psi.append(accu_attract_psi)
        attracts_latent_ids.append(latent_ids)
        vessels_psi[v, int(vessel_omega_group_id[v])] = vessel_psi_value

    attracts_psi = np.asarray(attracts_psi)
    attracts_latent_ids = np.asarray(attracts_latent_ids)
    features_psi = np.concatenate([attracts_psi, vessels_psi], axis=1)

    return features_psi, attracts_latent_ids


def psi_decay_horizon_greedy(goal_coords, vessels_coords, vessels_speeds, attract_flags, attract_sigmas, attract_omega, vessel_omega_group_id, disattract_sigma, lambda_=1.0):
    vessel_num = vessels_coords.shape[0]
    group_num = len(np.unique(vessel_omega_group_id))

    attract_latent_flags = np.zeros(attract_flags.shape)
    vessel_latent_flags = np.ones(vessel_num)
    vessel_flags = -1.0*np.ones(vessel_num)
    vessel_sigmas = disattract_sigma*np.ones(vessel_num)

    attracts_psi = []
    attracts_latent_ids = []
    vessels_psi = np.zeros((vessel_num, group_num))

    for v in range(len(vessels_coords)):
        accu_attract_psi = np.zeros(attract_flags.shape)
        latent_ids = []
        vessel_psi_value = 0.0

        for i in range(vessels_coords.shape[1]):
            vessel_coord = vessels_coords[v, i, :]
            vessel_speed = vessels_speeds[v, i, :]

            attract_force_magnitudes, attract_force_dirs = goal_force_euclidean(goal_coords, vessel_coord, attract_flags, attract_sigmas)

            max_step_log_posterior = -np.inf
            max_step_psi = np.zeros(attract_flags.shape)
            latent_id = -1
            for idx in range(len(attract_latent_flags)):
                new_latent_flags = attract_latent_flags
                new_latent_flags[idx] = 1.0
                tmp_psi = step_psi_with_selected_latent(attract_force_magnitudes, attract_force_dirs, vessel_speed, new_latent_flags)
                if attract_omega.dot(tmp_psi) > max_step_log_posterior:
                    max_step_log_posterior = attract_omega.dot(tmp_psi)
                    max_step_psi = tmp_psi
                    latent_id = idx

            accu_attract_psi += max_step_psi*lambda_**(vessels_coords.shape[1]-i)
            latent_ids.append(latent_id)
        
            vessel_force_magnitudes, vessel_force_dirs = goal_force_euclidean(vessels_coords[:, i, :], vessel_coord, vessel_flags, vessel_sigmas)
            vessel_psi = step_psi_with_selected_latent(vessel_force_magnitudes, vessel_force_dirs, vessel_speed, vessel_latent_flags)
            vessel_psi_value += np.sum(vessel_psi)*lambda_**(vessels_coords.shape[1]-i)

        attracts_psi.append(accu_attract_psi)
        attracts_latent_ids.append(latent_ids)
        vessels_psi[v, int(vessel_omega_group_id[v])] = vessel_psi_value

    attracts_psi = np.asarray(attracts_psi)
    attracts_latent_ids = np.asarray(attracts_latent_ids)
    features_psi = np.concatenate([attracts_psi, vessels_psi], axis=1)

    return features_psi, attracts_latent_ids