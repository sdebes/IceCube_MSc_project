"""
This script takes event files for MC and data and returns them with added photon reconstruction columns.
The added columns are:
doca
photon_distance
photon_z
photon_azimuth
photon_zenith
cherenkov_x
cherenkov_y
cherenkov_z
dist_cher_A
t_A

I get an error If I try to make a file with 25k+ events, so I use 20k events instead.
You can probably include more events by only saving the new columns with the event_no, and then merge afterwards but meh.
One file of 20k events takes about 1.5-2 hours.
Below you can select how many files of 20k events you want to generate.
"""

print('\nScript started.')
import numpy as np
import pandas as pd
import sqlite3
import time
start_time = time.time()
#############################################

N_events = int(80000)
N_files = 20

file_size = int(N_events/N_files)
N_events/file_size
print('N_events:', N_events)
print('file_size:', file_size)
print('N_files:', N_files)

output_folder = '/groups/icecube/debes/work/analyses/Efficiency/data/burnsample_efficiency_summed_charge_reco/'


rd_truth_path = "/groups/icecube/debes/storage/burnsample_predictions/14594307_stopped_muons_from_burnsample_with_pos_xyz_azimuth_zenith_reconstruction.parquet" #14.6m events from brunsample
mc_truth_weighted_path = "/groups/icecube/debes/storage/stopped_muons_final_reco_MC_GBweighted.csv" #8.3M events
rd_truth = pd.read_parquet(rd_truth_path)
mc_truth = pd.read_csv(mc_truth_weighted_path, nrows = N_events)

# DOM coordinates
all_dom_coordinates = pd.read_csv('/groups/icecube/debes/work/analyses/Efficiency/data/5083_DOM_coordinates_with_homemade_dom_id_from_burnsample.csv', index_col=0)

# Pulsemaps
rd_db_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
mc_db_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon2022_DataAndMC_CSVandDB_StoppedMuons/last_one_lvl3MC.db"
############################################

#Always selecting the lowest N_events event_nos in the files for reproducibility
rd_events = tuple(list(set(rd_truth['event_no'].sort_values().values[:N_events])))
mc_events = tuple(list(set(mc_truth['event_no'].sort_values().values[:N_events])))

############################################


def add_unactivated_doms(DataFrame, is_mc):
    Pulses = DataFrame.copy()
    mc_cols   = ['azimuth', 'zenith', 'position_x', 'position_y', 'position_z', 'GB_weights']
    columns   = ['event_no', 'charge', 'dom_time','dom_x', 'dom_y', 'dom_z', 'dom_id', 'width', 'azimuth_pred', 'zenith_pred', 'position_x_pred', 'position_y_pred', 'position_z_pred']
    fill_cols = ['azimuth_pred', 'zenith_pred', 'position_x_pred', 'position_y_pred', 'position_z_pred']
    if is_mc: # add weights
        fill_cols += mc_cols
        columns += mc_cols            
    
    dom_df = all_dom_coordinates.copy() # ALL DOMs
    output_size = len(dom_df)*file_size
    big_df = pd.DataFrame(index = range(output_size), columns = columns+['activated'])
    big_df_size = 0


    for event_no in [int(x) for x in Pulses['event_no'].unique()]: #For all events
        df = Pulses.loc[DataFrame['event_no'] == event_no, columns]
        df['charge'] = df.groupby(['dom_x', 'dom_y', 'dom_z'])['charge'].transform('sum')
        df = df.drop_duplicates(subset=['dom_x', 'dom_y', 'dom_z']) #drop duplicates???
        df = pd.merge(df, dom_df, how='right', on=['dom_x', 'dom_y', 'dom_z','dom_id'], indicator='activated')
            
        merged_df = pd.merge(df, dom_df, on=['dom_x', 'dom_y', 'dom_z'], how='outer', indicator=True)
        # Filter out rows that are present in both DataFrames
        not_in_both_df = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        df['activated'] = df['activated'].map({'right_only':0, 'both':1})
        df['event_no'] = event_no
        
        df[fill_cols] = df[fill_cols].fillna(method='ffill')

        num_rows = df.shape[0]
        big_df.iloc[big_df_size:big_df_size+num_rows] = df.values
        big_df_size += num_rows
    big_df = big_df.iloc[:big_df_size]  # Trim the excess unused rows
    big_df['event_no'] = big_df['event_no'].astype(int)
#    big_df['dom_id'] = big_df['dom_id'].astype(int)
    big_df[['event_no','dom_id', 'activated']] = big_df[['event_no','dom_id','activated']].astype(int)
    return big_df

def find_photon_distance(DataFrame, truth = False):
    df = DataFrame.copy()
    index_of_refraction = 1.32 # Value used by AMANDA
    cherenkov_angle = 40.75 * np.pi / 180 # = arccos(1/index_of_refraction)
    c = 299792458 * 1e-9 # m/ns

    if truth:
        end_points = df[['position_x', 'position_y', 'position_z']].values.astype(np.float32)
        azimuths   = df['azimuth'].values.astype(np.float32)
        zeniths    = df['zenith'].values.astype(np.float32)
    else:
        end_points = df[['position_x_pred', 'position_y_pred', 'position_z_pred']].values.astype(np.float32)
        azimuths   = df['azimuth_pred'].values.astype(np.float32)
        zeniths    = df['zenith_pred'].values.astype(np.float32)

    dom_points = df[['dom_x', 'dom_y', 'dom_z']].values.astype(np.float32)
    dom_times  = df['dom_time'].values.astype(np.float32)
    # Unit vectors in the direction the muon came from
    tracks = np.column_stack((np.cos(azimuths) * np.sin(zeniths),
                              np.sin(azimuths) * np.sin(zeniths),
                              np.cos(zeniths)))

    # Start points 1km away from the end points along the tracks
    start_points = end_points + 2000 * tracks # Are we sure it's + and not -?
    # Vectors going from the start points to the end points
    
    mu_vectors = end_points - start_points  # (R-Q)
    Q_P = start_points - dom_points
    # Some factors
    projection_factor = np.sum(mu_vectors * Q_P, axis=1) / np.sum(mu_vectors ** 2, axis=1)
    # Projection of the DOMs onto the tracks
    pDoca = start_points - projection_factor[:, np.newaxis] * mu_vectors 
    # Distance between the DOMs and the tracks
    dist_Doca_DOM = np.linalg.norm(dom_points - pDoca.astype(np.float32), axis=1) #
    dist_Doca_Cher = dist_Doca_DOM * np.tan(np.pi/2 - cherenkov_angle) #
    cherenkov_points = pDoca - np.expand_dims(dist_Doca_Cher, axis=1) * mu_vectors / np.expand_dims(np.linalg.norm(mu_vectors, axis=1), axis=1)

    photon_vector      = dom_points - cherenkov_points
    photon_vector_norm = photon_vector / np.linalg.norm(photon_vector, axis=1)[:, np.newaxis]
    photon_zeniths     = np.arccos(photon_vector_norm[:,2])
    photon_azimuths    = np.arctan2(photon_vector_norm[:,1], photon_vector_norm[:,0])
    photon_distances   = np.linalg.norm(dom_points - cherenkov_points, axis=1)
    dist_cher_A        = np.linalg.norm(end_points - cherenkov_points, axis=1)
    
    t_A = dom_times - photon_distances * index_of_refraction / c + dist_cher_A/c

    df['doca'] = dist_Doca_DOM
    df['photon_distance'] = photon_distances
    df['photon_z']        = np.mean([df['dom_z'], cherenkov_points[:,2]], axis=0)
    df['photon_azimuth']  = photon_azimuths
    df['photon_zenith']   = photon_zeniths
    df['cherenkov_x']     = cherenkov_points[:,0]
    df['cherenkov_y']     = cherenkov_points[:,1]
    df['cherenkov_z']     = cherenkov_points[:,2]
    df['dist_cher_A']     = dist_cher_A
    df['t_A'] = t_A
    return df

############################################
columns = ['charge', 'dom_time', 'dom_x', 'dom_y', 'dom_z', 'event_no', 'width']
for iteration, j in enumerate(np.linspace(0, N_events-file_size, N_files)):
    MIN, MAX = int(j), int(j+file_size)
#    if MAX <= 40000:
#        continue
    t0 = time.time()

    print('Processing file', iteration,'containing event', MIN, 'to', MAX)
    
    rd_selection = tuple([int(x) for x in list(rd_events)[MIN:MAX]])
    conn = sqlite3.connect(rd_db_path)
    rd_pulse = pd.read_sql_query("SELECT {} FROM SplitInIcePulses WHERE event_no in {}".format(", ".join(columns), rd_selection), conn)
    conn.close()

    mc_selection = tuple([int(x) for x in list(mc_events)[MIN:MAX]])
    conn = sqlite3.connect(mc_db_path)
    mc_pulse = pd.read_sql_query("SELECT {} FROM SplitInIcePulses WHERE event_no in {}".format(", ".join(columns), mc_selection), conn)
    conn.close()
    
    #Add dom_id to pulses
    rounding = 2
    rd_pulse[['dom_x', 'dom_y', 'dom_z']] = rd_pulse[['dom_x', 'dom_y', 'dom_z']].round(rounding)
    mc_pulse[['dom_x', 'dom_y', 'dom_z']] = mc_pulse[['dom_x', 'dom_y', 'dom_z']].round(rounding)
    rd_pulse = pd.merge(rd_pulse, all_dom_coordinates, on=['dom_x', 'dom_y', 'dom_z'], how='inner')
    mc_pulse = pd.merge(mc_pulse, all_dom_coordinates, on=['dom_x', 'dom_y', 'dom_z'], how='inner')

    mc = pd.merge(mc_pulse, mc_truth[['event_no', 'azimuth', 'azimuth_pred', 'zenith','zenith_pred','position_x','position_y', 'position_z', 'position_x_pred','position_y_pred', 'position_z_pred', 'GB_weights']], on=['event_no'], how='inner')
    rd = pd.merge(rd_pulse, rd_truth[['event_no', 'azimuth_pred', 'zenith_pred', 'position_x_pred', 'position_y_pred', 'position_z_pred']], on=['event_no'], how='inner')

    t001 = time.time()
    print('Events and pulsemaps loaded in:', int((t001-t0)/60), 'minutes.')

    N = str(int(file_size/1000))
    iteration_string = str(int(iteration))
    
    rd_with_unactivated_DOMs                     = add_unactivated_doms(rd, is_mc = False)
    print('rd_with_unactivated_DOMs.shape:',rd_with_unactivated_DOMs.shape)
    rd_with_unactivated_DOMs_and_photon_distance = find_photon_distance(rd_with_unactivated_DOMs, truth = False)
    print('rd_with_unactivated_DOMs_and_photon_distance.shape:',rd_with_unactivated_DOMs_and_photon_distance.shape)
    rd_with_unactivated_DOMs_and_photon_distance.to_parquet(output_folder + f'{N}k_rd_photon_efficiency_reco_summed_charge_{iteration_string}.parquet')
    t1 = time.time()
    print('Data file completed in:', int((t001-t0)/60), 'minutes.')

    mc_with_unactivated_DOMs                      = add_unactivated_doms(mc, is_mc = True)
    mc_with_unactivated_DOMs_and_photon_distances = find_photon_distance(mc_with_unactivated_DOMs, truth = False)
    mc_with_unactivated_DOMs_and_photon_distances.to_parquet(output_folder + f'{N}k_mc_photon_efficiency_reco_summed_charge_{iteration_string}.parquet')
    t2 = time.time()
    print('MC file completed in:', int((t2-t1)/60), 'minutes.')
    
#    mct_with_unactivated_DOMs                      = add_unactivated_doms(mc, is_mc = True)
    mct_with_unactivated_DOMs_and_photon_distances = find_photon_distance(mc_with_unactivated_DOMs, truth=True)
    mct_with_unactivated_DOMs_and_photon_distances.to_parquet(output_folder + f'{N}k_mc_truth_photon_efficiency_reco_summed_charge_{iteration_string}.parquet')
    t3 = time.time()
    print('MC truth file completed in:', int((t3-t2)/60), 'minutes.')
    
    print('Data file has the shape', rd_with_unactivated_DOMs_and_photon_distance.shape)
    print('MC files have the shape', mc_with_unactivated_DOMs_and_photon_distances.shape)
    print('Time used for the photon reconstruction of event', MIN,'to', MAX, ':', int((t3-t0)/60), 'minutes.')

end_time = time.time()
print('Script completed in {} minutes.'.format(int((end_time-start_time)/60)))