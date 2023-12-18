from diffusion_policy.gaussians_utils.panorama_cam import read_and_dump_json, read_and_dump_json_auto_split


# # static, train = panoramic + wrist
# json_file_list = ['/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_panorama_move_fov_45_static_wrist',
#                   '/local/real/chuerpan/repo/4DGaussians/data/robomimic_old/can_panorama_move_fov_45_static_200_01']
# json_save_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_fov_45_static_train_pano_wrist_test_wrist'
# read_and_dump_json(json_file_list= json_file_list, json_save_path = json_save_path, stage = 'train')


 
# # static, train/test = panoramic + wrist, put wrist as the second path in the list
# json_file_list = ['/local/real/chuerpan/repo/4DGaussians/data/robomimic_old/can_panorama_move_fov_45_static_200_01',
#                   '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_panorama_move_fov_45_static_wrist']
# json_save_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_fov_45_static_train_pano_wrist_test_wrist_all'

# read_and_dump_json_auto_split(json_file_list= json_file_list, json_save_path = json_save_path, test_percentage=0.1)


# # static panoramic, dynamic wrist, train/test = panoramic + wrist, put wrist as the second path in the list
# json_file_list = ['/local/real/chuerpan/repo/4DGaussians/data/robomimic_old/can_panorama_move_fov_45_static_200_01',
#                   '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_main_wrist_fov_45_sequencial']
# json_save_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_fov_45_dynamic_train_pano_wrist_test_wrist_all'

# read_and_dump_json_auto_split(json_file_list= json_file_list, json_save_path = json_save_path, test_percentage=0.1)



# static panoramic, dynamic wrist, FOV = 90, train/test = panoramic + wrist, put wrist as the second path in the list
json_file_list = ['/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_panorama_move_fov_90_static_200',
                  '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_main_wrist_dynamic_fov_90_sequencial']
json_save_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_fov_90_dynamic_train_pano_wrist_test_wrist_all'

read_and_dump_json_auto_split(json_file_list= json_file_list, json_save_path = json_save_path, test_percentage=0.1)