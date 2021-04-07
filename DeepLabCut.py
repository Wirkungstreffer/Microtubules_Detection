import deeplabcut
deeplabcut.create_new_project('Microtubules','zixuan',['/home/ben/Microtubules_Detection/200818_xb_reaction2_6um0003.avi'])
path_config = '/home/ben/Microtubules_Detection/Microtubules-zixuan-2021-04-06/config.yaml'
deeplabcut.extract_frames(path_config, 'automatic','kmeans')
deeplabcut.label_frames(path_config)