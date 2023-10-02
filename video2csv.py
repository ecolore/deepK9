# -*- coding: utf-8 -*-
"""

@author: joseph-2023-10-01

 Description:
  1.) use Deeplabcut to analyze video and to generate .csv file
  2.) use Deeplabcut to generate labeled video
  3.) Normalized all .csv dataset and put all into output_csv_path

  Usage:
    optional arguments:
    -h, --help        show this help message and exit
    --gen_video       Generate labeled video.
    --gen_train_data  Generate training dataset with normalization.
    --new             Create new project config
   
  > python video2csv.py --gen_video --gen_train_data
  > python video2csv.py --gen_train_data
  > python video2csv.py -h
  > python video2csv.py --gen_train_data --new
 


"""

import os
import argparse
import shutil
from normalized import csv_std
import deeplabcut
import glob
import inference

#global 
new_project = False  

def list_files_recursive(path):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files

def analyze_video(basedir):
    files = list_files_recursive(basedir)
    for file in files:
        print(file)
        deeplabcut.analyze_videos(config_path,[file],save_as_csv=True)
            
        
        if gen_labeld_video==True :
            print("Generate labeled video")
        
            #it will create a new xxx_filtered.csv from a unfiltered .csv in videos dirddtory
            deeplabcut.filterpredictions(config_path, [file])
           
            # it will create a labeled video for each video file according the xxx_filtered.csv 
            deeplabcut.create_labeled_video(config_path, [file], filtered=True)


def copy_csv_files_to_folder(source_folder, target_folder):

    print('Copy all .csv files from {} to {}'.format(source_folder,target_folder))
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in os.listdir(source_folder):
        if filename.endswith(".csv") and 'filtered' not in filename:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(target_folder, filename))


def gen_training_data(basedir):
    
    files = list_files_recursive(basedir)
    for unfilter_csv in files:
        if '_normalized' in unfilter_csv: return 
        print(unfilter_csv)
        prefix=os.path.dirname(unfilter_csv)
        filename=os.path.basename(unfilter_csv)
        name,ext=os.path.splitext(filename)
        new_name=os.path.join(prefix,name+'_normalized'+ext)
        print(new_name)

        std_csv = csv_std(unfilter_csv)
        std_data=std_csv.auto_run(new_name)

def crea1proj():
    global new_project
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
    else:
      config_path=os.path.join(project_path,'config.yaml')
        
    return config_path  
   

def main(args):

    global gen_labeld_video,gen_training_data,new_project

    if args.new:
      new_project=True
   
    if args.gen_video:
       
        print("Generate labeled video")
        gen_labeld_video=True
        analyze_video(data_path)

    if args.gen_train_data:
       
        copy_csv_files_to_folder(video_source_folder,output_csv_folder)
        
        gen_training_data(output_csv_folder)
        
 
if __name__ == "__main__":

    gen_labeld_video=False
    video_source_folder='myvideos'
    output_csv_folder = 'csv_dataset'

    project_path=os.path.dirname(os.path.abspath(__file__)) #r'E:\DeepLabCut\topic-joseph-2023-10-01'
    print('project_path=',project_path)
    #project path and config_path
    config_path=crea1proj()

    data_path=os.path.join(os.getcwd(),video_source_folder)
    output_csv_path=os.path.join(os.getcwd(),output_csv_folder)

    csv_files = glob.glob(output_csv_path + '/*000.csv')

    if csv_files:
      newest_csv = max(csv_files, key=os.path.getmtime)
      print("Newest CSV file:", newest_csv)
    
    
#    print('project_path:',project_path)
    #print('config_path:',config_path)
    print('video_source_folder:',data_path)
    print('output_csv_folder:',output_csv_path)

    parser = argparse.ArgumentParser(description="Analyze the video ")

    # Boolean flag for gen_video 
    parser.add_argument("--gen_video", action="store_true", help="Generate labeled video.")
    # Boolean flag for gen training dataset with normalization 
    parser.add_argument("--gen_train_data", action="store_true", help="Generate training dataset with normalization.")
    parser.add_argument("--new",action="store_true", help="Create project config")

    args = parser.parse_args()
    
    main(args)

    ######## next step: inference
#    s_name=f'mydog123DLC_resnet50_topicOct1shuffle1_75000.csv'
 #   unfilter_csv = os.path.join(video_path,s_name)
    std_csv = csv_std(newest_csv)
    std_data=std_csv.auto_md()
    print(std_data.head())
    cls,conf=inference.predict(std_data)
    print('Predict {} (conf:{:.2f})'.format(cls,conf))

     