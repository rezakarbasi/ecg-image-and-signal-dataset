import pandas as pd
import numpy as np
import wfdb
import ast
import json
import matplotlib.pyplot as plt
import os
import sys
from math import ceil
import shutil
# sys.path.append("./prepare_image_dataset/adjusted_library")  # in order to import the adjusted "ecg_plot" library

from utils import determine_bb, crop_bb, plot_v3, get_indices_information, save_as_jpg
from dotenv import load_dotenv
import cv2

load_dotenv()



# def bw_to_yolo(file_path, export_path):
#   img = cv2.imread(file_path)

#   kernel = np.ones((2,2),np.uint8)
#   dilated = cv2.dilate(img,kernel,iterations = 1)

#   # Threshold the image to get a binary mask
#   _, binary_mask = cv2.threshold(np.uint8(dilated[:,:,0]), 127, 255, cv2.THRESH_BINARY)

#   # Find contours
#   contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   contour = sorted(contours, key = lambda x: -cv2.contourArea(x))[0]
#   contour = contour.squeeze(1)

#   height, width = img.shape[:2]

#   output = ["0"]
#   for i in contour:
#     output.append(f"{i[0]/width:0.04f}")
#     output.append(f"{i[1]/height:0.04f}")

#   output = " ".join(output)

#   with open(export_path, 'w') as f:
#     f.write(output)

sampling_rate = 100

# def load_raw_signal()
# if the .npy file of signal is not available, we will make it; otherwise, it will be loaded
if not os.path.exists(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy'):  
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    path = os.getenv("raw_signal_path")

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    np.save(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy', X)
    print(f'signal dataset was saved into all_signals_{sampling_rate}Hz.npy file')
else:
    X = np.load(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy')
    print(f'all_signals_{sampling_rate}Hz.npy file loaded')

dpi = 700
### convert the signal to ecg image
dataset_version = 8

# sample each lead for train, validation, and test sets
sample_each_lead_train = 5
sample_each_lead_val = 2
sample_each_lead_test = 2

padding_x:int = 30        # it's the padding around the bounding boxes
padding_y:int = 30        # it's the padding around the bounding boxes
border:int = 34         # it's the white border in down left of the image that should be considered in determining the bounding box
# for dpi: border -> 250: 12, 700: 34

dataset_path = f'{os.getenv("datasets_path")}image_dataset_v{dataset_version}.0'

path_to_detection_dataset = f'{dataset_path}/detection/'
path_to_segmentation_dataset = f'{dataset_path}/segmentation/'
path_to_temp_dir = f'{dataset_path}/temp/'

for p in [path_to_detection_dataset, path_to_segmentation_dataset, path_to_temp_dir]:
    if os.path.exists(p):
        shutil.rmtree(p)   # remove the directory

os.makedirs(path_to_detection_dataset)
os.makedirs(path_to_segmentation_dataset)
os.makedirs(path_to_temp_dir)

# Create child directories: test, train, val
for folder in ["train", "test", "val"]:
    os.makedirs(os.path.join(path_to_segmentation_dataset, folder))
    
    os.makedirs(os.path.join(path_to_detection_dataset, folder, "images"))
    os.makedirs(os.path.join(path_to_detection_dataset, folder, "labels"))

    os.makedirs(os.path.join(path_to_segmentation_dataset, folder, "image"))
    os.makedirs(os.path.join(path_to_segmentation_dataset, folder, "mask-png"))
    os.makedirs(os.path.join(path_to_segmentation_dataset, folder, "mask-bmp"))
    os.makedirs(os.path.join(path_to_segmentation_dataset, folder, "signal-json"))

print(f'directory {path_to_detection_dataset} created.')
print(f'directory {path_to_segmentation_dataset} created.')


row_height = 3
lead_index = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # order of the leads in the dataset
lead_display = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # order of the lead that I want to be shown
lead_config = {
                '3by1': {'n_column':1, 'length': 1000, 'lead_order': list(range(3)), 'full_ecg_name': None, 'n_leads': 3}, 
                '3by4': {'n_column':4, 'length': 250, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 
                        'full_ecg_name': None,
                        'n_leads': 12},
                # '12by1':  {'n_column':1, 'length': 1000, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': None, 'n_leads': 12}, 
                # '6by2': {'n_column':2, 'length': 500, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': 'II', 'n_leads': 12}, 
            }  # key determines lead format, value determines some variable passing to ecg_plot_vs.plot
#TODO: the above lead_config should be uncommented. I wanted to create dataset containing only one lead format for initial development of UNet

train_indices, end_idx = get_indices_information(0, lead_config, sample_each_lead_train)
val_indices, _ = get_indices_information(end_idx, lead_config, sample_each_lead_val)
test_indices, _ = get_indices_information(X.shape[0]-sample_each_lead_test*len(lead_config), lead_config, sample_each_lead_test)

logs = {'train': [], 'test': [], 'val': []}

for data_pack, dtset_type in [(train_indices, "train"), (test_indices, "test"), (val_indices, "val")]:
    for idx, lead_format, each_lead_config in data_pack:
        print(dtset_type, idx)
        sample_name = str(idx)+'_'+lead_format
        detection_export_path = os.path.join(path_to_detection_dataset, dtset_type)
        segmentation_export_path = os.path.join(path_to_segmentation_dataset, dtset_type)

        plt.close("all")

        image_temp_path = path_to_temp_dir
        blackwhite_path = os.path.join(image_temp_path, sample_name+"-bw.jpg")

        detection_input_path = os.path.join(detection_export_path, "images")
        detection_output_path = os.path.join(detection_export_path, "labels")

        segmentation_image_path = os.path.join(segmentation_export_path, "image")
        segmentation_mask_png_path = os.path.join(segmentation_export_path, "mask-png")
        segmentation_mask_bmp_path = os.path.join(segmentation_export_path, "mask-bmp")
        segmentation_signal_path = os.path.join(segmentation_export_path, "signal-json")


        image_detection_sample = os.path.join(detection_input_path, sample_name+".jpg")
        label_detection_sample = os.path.join(detection_output_path, sample_name+".txt")

        # image_segmentation_sample = os.path.join(segmentation_image_path, sample_name+".jpg")
        mask_png_segmentation_sample = os.path.join(segmentation_mask_png_path, sample_name+".png")
        mask_bmp_segmentation_sample = os.path.join(segmentation_mask_bmp_path, sample_name+".bmp")
        signal_segmentation_sample = os.path.join(segmentation_signal_path, sample_name+".json")

        log = plot_v3(
            ecg=X[idx, :each_lead_config['length'], :each_lead_config['n_leads']].T,
            full_ecg=X[idx, :, 1].T, 
            full_ecg_name=each_lead_config['full_ecg_name'],
            sample_rate=sampling_rate, 
            columns=each_lead_config['n_column'],
            lead_index=lead_index,
            title='', 
            lead_order=each_lead_config['lead_order'],
            show_lead_name=False,
            show_grid=True, 
            show_separate_line=False,
            row_height=row_height,
            style=None,
            save_path=image_detection_sample,
            dpi=dpi)

        log['padding_x'] = padding_x
        log['padding_y'] = padding_y
        log['border'] = border

        # convert the ploted image to an np.array in order to have the image array
        fig = plt.gcf()
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)
        save_as_jpg(image_detection_sample, dpi=dpi)

        crop_bb(sample=log, export_path=segmentation_image_path, img_path=image_detection_sample, prefix=sample_name+"_")
        determine_bb(sample=log, mode='online', save_bb_path=label_detection_sample, img_array=image_array)

        for lead_number in range(each_lead_config['n_leads']+1):
            if lead_number==each_lead_config['n_leads']:
                inserted_signal = np.zeros_like(X[idx, :each_lead_config['length'], :each_lead_config['n_leads']])
                inserted_full = X[idx, :, 1]

                if isinstance(each_lead_config['full_ecg_name'], type(None)):
                    continue
            else:
                inserted_signal = np.zeros_like(X[idx, :each_lead_config['length'], :each_lead_config['n_leads']])
                inserted_signal[:,lead_number] = X[idx, :each_lead_config['length'], lead_number]

                inserted_full = np.zeros_like(X[idx, :, 1])

            plot_v3(
                ecg=inserted_signal.T,
                full_ecg=inserted_full.T, 
                full_ecg_name=each_lead_config['full_ecg_name'],
                sample_rate=sampling_rate, 
                columns=each_lead_config['n_column'],
                lead_index=lead_index,
                title='', 
                lead_order=each_lead_config['lead_order'],
                show_lead_name=False,
                show_grid=False, 
                show_separate_line=False,
                row_height=row_height,
                style='binary',
                save_path=blackwhite_path,
                dpi=dpi)

            fig = plt.gcf()
            fig.canvas.draw()
            bw_array = np.array(fig.canvas.renderer._renderer)
            save_as_jpg(blackwhite_path, dpi=dpi)

            crop_bb(sample=log, export_path=segmentation_mask_bmp_path, img_path=blackwhite_path, prefix=sample_name+"_", smaple_number=lead_number, save_bmp=True)
            crop_bb(sample=log, export_path=segmentation_mask_png_path, img_path=blackwhite_path, prefix=sample_name+"_", smaple_number=lead_number)

            with open(os.path.join(segmentation_signal_path, f"{sample_name}_lead_{lead_number}.json"), 'w') as f:
                json.dump(log['leads'][lead_number]['ecg'], f)
            
            # bw_to_yolo(os.path.join(digitization_input_path, f"{sample_name}_lead_{lead_number}.png"), os.path.join(segmentation_output_path, f"{sample_name}_lead_{lead_number}.txt"))

        logs[dtset_type].append(log)

for dataset_type, value in logs.items():
    with open(os.path.join(dataset_path, f"logs_segmentation_{dataset_type}.json"), 'w') as f:
        json.dump({"frequency": sampling_rate, "samples":value}, f)
