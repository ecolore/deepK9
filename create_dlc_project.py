# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:03:06 2023

@author: joseph@艾鍗學院

'video2csv.py' : it can be used to generate the training dataset .

 
   1.mp4--> 1.csv
   2.mp4--> 2.csv
    ...
   3.mp4--> 3.csv


https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md


pip install deeplabcut[gui]


"""

import os
import numpy as np
import pandas as pd
import deeplabcut


training=False  #set False if you only want to generate .csv file 
new_project=True
add_new_video=False

base_dir=os.path.dirname(os.path.abspath(__file__))

def crea1proj():
    project_path=base_dir #r'D:\Deep-Jose'
    video_source_folder=os.path.join(project_path,'myvideos')
    #config_path=os.path.join(project_path,'config.yaml')
    video_path= os.path.join(project_path,'myvideos')

    print('project_path:',project_path)

    print('video_path:',video_path)


    if __name__ == '__main__':  #avoid spawn infinitely

        # Step 1:
        # Create a new pre-trained project directory and return a configuration file (.yaml) in it.
        # When creating new proejct , it will create .csv file, labeled video and some others file
        # for each .mp4 file in ./video directory. 
        # The model 'full_dog' will be downloaded and put in dlc-models diectory automaticlly.

        if new_project==True:
        
            config_path, train_config_path = deeplabcut.create_pretrained_project(
            'topic',
            'joseph',
            videos=[video_path],  # Create the labeled video for all the videos with an .mp4 extension in a directory.
            videotype='.mp4',
            model="full_dog",
            analyzevideo=True,
            createlabeledvideo=True,
            copy_videos=True
            )
    return config_path
    
#config_path=crea1proj()
#print('config_path:',config_path)


#Add a new video to config.yaml
if add_new_video==True:
#        base_dir=os.path.dirname(os.path.abspath(__file__))
    myvideo=os.path.join(base_dir,"mydog123.mp4")
    deeplabcut.add_new_videos(config_path,[myvideo], copy_videos=True)



# Step 2:
# The project_path need to modify manually to fit your situation
# Edit the config.ymal file to configure your project
    '''
    edits = {
    'dotsize': 1,
    'colormap': 'spring',
    'pcutoff': 0.5,
    }

    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)
    '''

    if training==True:
        
        '''
          Below is to customiz/fine-tune the network 
        '''

        # Step 3: Extract video frames to annotate
        deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop=True)

        # Step 4: Annotate all the extracted frames using an interactive GUI
        deeplabcut.label_frames(config_path)

        # Step 5: creates a subdirectory with labeled as a suffix. 
        #Those directories contain the frames plotted with the annotated body parts

        # Step 6: create_training_dataset
        deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

        # Step 7:train the network
        deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, 
        gputouse=None, max_snapshots_to_keep=5, autotune=False, displayiters=100, saveiters=15000, maxiters=30000, allow_growth=True)

        # Step ....
        #deeplabcut.check_labels(config_path, visualizeindividuals=True)
        #deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)
        #deeplabcut.extract_save_all_maps(config_path, shuffle=shuffle, Indices=[0, 5])
         
        #it will create a new xxx_filtered.csv from a unfiltered .csv in videos dirddtory
        deeplabcut.filterpredictions(config_path, video_path)


        # it will create a labeled video for each video file according the xxx_filtered.csv 
        deeplabcut.create_labeled_video(config_path, [video_path], filtered=True)


