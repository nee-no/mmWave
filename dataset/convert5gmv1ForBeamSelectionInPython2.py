''' 
Author : Aldebaro Klautau, Pedro Batista, Nuria Gonzalez-Prelcic, Yuyang Wang and Robert W. 
Date: 2018 
Title: 5gm-data/convert5gmv1ForBeamSelectionInPython.py 
Code Version: 
Availability: https://github.com/lasseufpa/5gm-data/blob/master/ convert5gmv1ForBeamSelectionInPython.py 
Modified by : Shalom Lee
''' 
#Script to generate data for beam selection using only the position of vehicles. 
#The output npz file has six arrays:
#the first (position_matrix_array) is the input of machine learning algorithms &
#informs the vehicles positions;
#the second (best_ray_array) is the output and can represent two alternatives,
#depending on the variable use_geometricMIMOChannelModel.
#the third (path_gains_array), fourth (departure_angles_array) & fifth (arrival_angles_array) are used in calculation of H, the channel matrix
#the sixth (t1s_array) is the received power and used in reinforcement learning


import datetime
import numpy as np
from shapely import geometry

from rwisimulation.positionmatrix import position_matrix_per_object_shape, calc_position_matrix
from rwisimulation.calcrxpower import calc_rx_power

from rwisimulation.datamodel import save5gmdata as fgdb

#import config as c
class c: #this information is obtained from the config.py file used to generate the data
    #analysis_area = (648, 348, 850, 685)
    analysis_area = (744, 429, 767, 679) #coordinates that define the areas the mobile objects should be
    analysis_area_resolution = 0.5 #grid resolution in meters
    antenna_number = 4 #number of antenna elements in Rx array
    frequency = 6e10 #carrier frequency in Hz

analysis_polygon = geometry.Polygon([(c.analysis_area[0], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[3]),
                                     (c.analysis_area[0], c.analysis_area[3])])

only_los = True #use or not only the Line of Sight (LOS) channels

#use_geometricMIMOChannelModel determines what is written in the output array best_ray_array
#For "classification", use True. For "regression", use False
#If True, the output are 2 numbers: the best Tx and Rx codebook indices
#If False, the output are 4 real numbers, representing the angles (azimuth and elevation) for Tx and Rx
#The array best_ray_array is later used as the output of e.g. neural networks
use_geometricMIMOChannelModel = True

#npz_name = 'episode.npz' #output file name

session = fgdb.Session()

pm_per_object_shape = position_matrix_per_object_shape(c.analysis_area, c.analysis_area_resolution)
#print(pm_per_object_shape)

# do not look, just to report
start = datetime.datetime.today()
perc_done = None
num_scene = 1

totalNumEpisodes = session.query(fgdb.Episode).count()
print('Found ', totalNumEpisodes, ' episodes. Processing...')
for ep in session.query(fgdb.Episode):
    npz_name = 'episode' + str(num_scene) + '.npz'
    num_scene += 1
    # 50 scenes, 10 receivers per scene
    print('Processing ', ep.number_of_scenes, ' scenes in episode ', ep.insite_pah,)
    print('Start time = ', ep.simulation_time_begin, ' and sampling period = ', ep.sampling_time, ' seconds')
    #Assumes 50 scenes per episode and 10 Tx/Rx pairs per scene
    position_matrix_array = np.zeros((50, 10, *pm_per_object_shape), np.int8)
    if use_geometricMIMOChannelModel:
        best_ray_array = np.zeros((50, 10, 2), np.float32) #2 numbers are the best Tx and Rx codebook indices
    else:
        best_ray_array = np.zeros((50, 10, 4), np.float32) #4 angles (azimuth and elevation) for Tx and Rx
        
    path_gains_array = np.zeros((50, 10, 25), np.float32)
    departure_angles_array = np.zeros((50, 10, 25, 2), np.float32)
    arrival_angles_array = np.zeros((50, 10, 25, 2), np.float32)
    t1s_array = np.zeros((50, 10, np.power(c.antenna_number, 2), np.power(c.antenna_number, 2)), np.float32)
    
    best_ray_array.fill(np.nan)
    rec_name_to_array_idx_map = [obj.name for obj in ep.scenes[0].objects if len(obj.receivers) > 0]
    print(rec_name_to_array_idx_map)
    for sc_i, sc in enumerate(ep.scenes):
        polygon_list = []
        polygon_z = []
        polygons_of_interest_idx_list = []
        rec_present = []
        for obj in sc.objects:
            obj_polygon = geometry.asMultiPoint(obj.vertice_array[:,(0,1)]).convex_hull
            # check if object is inside the analysis_area
            if obj_polygon.within(analysis_polygon):
                # if the object is a receiver calc a position_matrix for it
                if len(obj.receivers) > 0:
                    rec_array_idx = rec_name_to_array_idx_map.index(obj.name)
                    for rec in obj.receivers:
                        best_ray = None
                        best_path_gain = - np.inf
                        for ray in rec.rays:
                            if ray.path_gain > best_path_gain:
                                best_path_gain = ray.path_gain
                                best_ray = ray
                        if (best_ray is not None and not best_ray.is_los) or not only_los:
                            if use_geometricMIMOChannelModel:
                                departure_angle_array = np.empty((len(rec.rays), 2), np.float64)
                                arrival_angle_array = np.empty((len(rec.rays), 2), np.float64)
                                p_gain_array = np.empty((len(rec.rays)), np.float64)
                                
                                for ray_i, ray in enumerate(rec.rays):
                                    departure_angle_array[ray_i, :] = np.array((
                                        ray.departure_elevation,
                                        ray.departure_azimuth,
                                    ))
                                    
                                    arrival_angle_array[ray_i, :] = np.array((
                                        ray.arrival_elevation,
                                        ray.arrival_azimuth,
                                    ))
                                    p_gain_array[ray_i] = np.array((ray.path_gain))
                                    
                                #from IPython import embed
                                #embed()
                                t1 = calc_rx_power(departure_angle_array, arrival_angle_array, p_gain_array, c.antenna_number, c.frequency)
                                
                                t1_abs = np.abs(t1)
                                t1s_array[sc_i, rec_array_idx, :, :] = t1_abs
                                path_gain = p_gain_array.reshape(-1)
                                path_gains_array[sc_i, rec_array_idx, :] = path_gain
                                departure_angles_array[sc_i, rec_array_idx, :, :] = departure_angle_array
                                arrival_angles_array[sc_i, rec_array_idx, :, :] = arrival_angle_array
                                                   
                                
                                best_ray_array[sc_i, rec_array_idx, :] = \
                                    np.argwhere(t1_abs == np.max(t1_abs))
                            else:
                                best_ray_array[sc_i, rec_array_idx, :] = np.array((
                                    best_ray.departure_elevation,
                                    best_ray.departure_azimuth,
                                    best_ray.arrival_elevation,
                                    best_ray.arrival_azimuth))
                    if (best_ray is not None and not best_ray.is_los) or not only_los:
                        # the next polygon added will be the receiver
                        polygons_of_interest_idx_list.append(len(polygon_list))
                        rec_present.append(obj.name)
                polygon_list.append(obj_polygon)
                polygon_z.append(-obj.dimension[2])
        if len(polygons_of_interest_idx_list) != 0:
            scene_position_matrix = calc_position_matrix(
                c.analysis_area,
                polygon_list,
                c.analysis_area_resolution,
                polygons_of_interest_idx_list,
                polygon_z=polygon_z,
            )
        for rec_i, rec_name in enumerate(rec_present):
            rec_array_idx = rec_name_to_array_idx_map.index(rec_name)
            position_matrix_array[sc_i, rec_array_idx, :] = scene_position_matrix[rec_i]

        # just reporting spent time
        perc_done = ((sc_i + 1) / ep.number_of_scenes) * 100
        elapsed_time = datetime.datetime.today() - start
        time_p_perc = elapsed_time / perc_done

        print('\r Done: {:.2f}% Scene: {} time per scene: {} time to finish: {}'.format(
            perc_done,
            sc_i + 1,
            elapsed_time / (sc_i + 1),
            time_p_perc * (100 - perc_done)), end='')
        
    print()
    np.savez(npz_name, position_matrix_array=position_matrix_array,
             best_ray_array=best_ray_array, path_gains_array=path_gains_array, departure_angles_array=departure_angles_array, arrival_angles_array=arrival_angles_array, t1s_array=t1s_array)
    print('Saved file ', npz_name)

#save output file with two arrays
#np.savez(npz_name, position_matrix_array=position_matrix_array,
#             best_ray_array=best_ray_array)
#print('Saved file ', npz_name)
