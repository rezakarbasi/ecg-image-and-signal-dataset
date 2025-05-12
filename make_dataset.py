"""
make_dataset.py

This script generates image and signal datasets from raw ECG data for detection and segmentation tasks.
It loads ECG signals, processes them into images and masks, and organizes them into train/val/test splits.
The script is highly configurable via environment variables or command-line arguments.

Usage:
    python make_dataset.py [--sample_each_lead_train N] [--sample_each_lead_val N] [--sample_each_lead_test N] ...
See argparse section below for all options.
"""

import pandas as pd
import numpy as np
import wfdb
import ast
import json
import matplotlib.pyplot as plt
import os
import shutil
import argparse
from tqdm import tqdm

from utils import generate_bounding_boxes, crop_lead_images, plot_ecg_multilead, generate_split_indices, save_figure_as_jpg
from dotenv import load_dotenv

# ---------------------- Argument Parsing ----------------------
def parse_args():
    """
    Parse command-line arguments and environment variables for dataset generation.

    Arguments can be provided via the command line or .env file.
    Priority: command-line > environment variable > default.

    Key arguments:
        --sample_each_lead_train: Number of samples per lead for training set.
        --sample_each_lead_val:   Number of samples per lead for validation set.
        --sample_each_lead_test:  Number of samples per lead for test set.
        --padding_x, --padding_y: Padding for bounding boxes.
        --border:                 Border size for images.
        --row_height:             Height of each ECG row in the image.
        --signal_dataset_path:    Path to save/load processed signal arrays.
        --raw_signal_path:        Path to raw ECG files.
        --datasets_path:          Output directory for generated datasets.
    """
    parser = argparse.ArgumentParser(description="Generate ECG image and signal datasets for detection/segmentation tasks.")
    parser.add_argument('--sample_each_lead_train', type=int, default=get_env_int("SAMPLE_EACH_LEAD_TRAIN", 5),
                        help="Number of samples per lead for training set.")
    parser.add_argument('--sample_each_lead_val', type=int, default=get_env_int("SAMPLE_EACH_LEAD_VAL", 2),
                        help="Number of samples per lead for validation set.")
    parser.add_argument('--sample_each_lead_test', type=int, default=get_env_int("SAMPLE_EACH_LEAD_TEST", 2),
                        help="Number of samples per lead for test set.")
    parser.add_argument('--padding_x', type=int, default=get_env_int("PADDING_X", 30),
                        help="Horizontal padding for bounding boxes.")
    parser.add_argument('--padding_y', type=int, default=get_env_int("PADDING_Y", 30),
                        help="Vertical padding for bounding boxes.")
    parser.add_argument('--border', type=int, default=get_env_int("BORDER", 34),
                        help="Border size for images.")
    parser.add_argument('--row_height', type=int, default=get_env_int("ROW_HEIGHT", 3),
                        help="Height of each ECG row in the image.")
    parser.add_argument('--signal_dataset_path', type=str, default=get_env_str("SIGNAL_DATASET_PATH", "data/signals"),
                        help="Path to save/load processed signal arrays.")
    parser.add_argument('--raw_signal_path', type=str, default=get_env_str("RAW_SIGNAL_PATH", "data/raw"),
                        help="Path to raw ECG files.")
    parser.add_argument('--datasets_path', type=str, default=get_env_str("DATASETS_PATH", "data/datasets"),
                        help="Output directory for generated datasets.")
    return parser.parse_args()

# ---------------------- Utility Functions ----------------------
def get_env_int(var, default):
    """Get integer environment variable or default."""
    return int(os.getenv(var, default))

def get_env_str(var, default):
    """Get string environment variable or default."""
    return os.getenv(var, default)

def get_env_list(var, default):
    """Get list from comma-separated environment variable or default."""
    return os.getenv(var, default).split(",")

def load_raw_data(metadata_df, sampling_rate, raw_path):
    """
    Load raw ECG signal data from files using wfdb.
    Returns a numpy array of shape (num_samples, signal_length, num_leads).
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(raw_path, f)) for f in metadata_df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(raw_path, f)) for f in metadata_df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(annotation_dict, agg_df):
    """
    Aggregate diagnostic classes from the annotation dictionary.
    Returns a list of unique diagnostic classes for a sample.
    """
    classes = []
    for key in annotation_dict.keys():
        if key in agg_df.index:
            classes.append(agg_df.loc[key].diagnostic_class)
    return list(set(classes))

def prepare_directories(dataset_root, detection_dir, segmentation_dir, temp_dir):
    """
    Remove and recreate dataset directories for detection and segmentation tasks.
    """
    for p in [detection_dir, segmentation_dir, temp_dir]:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)
    for split in ["train", "test", "val"]:
        os.makedirs(os.path.join(segmentation_dir, split))
        os.makedirs(os.path.join(detection_dir, split, "images"))
        os.makedirs(os.path.join(detection_dir, split, "labels"))
        os.makedirs(os.path.join(segmentation_dir, split, "image"))
        os.makedirs(os.path.join(segmentation_dir, split, "mask-png"))
        os.makedirs(os.path.join(segmentation_dir, split, "mask-bmp"))
        os.makedirs(os.path.join(segmentation_dir, split, "signal-json"))

def process_sample(
    sample_idx, lead_format, lead_cfg, signals, sampling_rate, row_height, 
    lead_names, detection_dir, segmentation_dir, temp_dir, 
    padding_x, padding_y, border, dpi
):
    """
    Process a single ECG sample: generate images, masks, bounding boxes, and signal JSONs.

    Args:
        sample_idx: Index of the sample in the signals array.
        lead_format: Format key (e.g., '3by1', '3by4').
        lead_cfg: Configuration dict for the lead format.
        signals: Numpy array of ECG signals.
        sampling_rate: Sampling rate of the signals.
        row_height: Height of each ECG row in the image.
        lead_names: List of lead names.
        detection_dir: Output path for detection images/labels.
        segmentation_dir: Output path for segmentation images/masks.
        temp_dir: Temporary directory for intermediate files.
        padding_x, padding_y, border, dpi: Image and bounding box parameters.

    Returns:
        log: Dictionary with plotting and bounding box info for this sample.
    """
    sample_name = f"{sample_idx}_{lead_format}"
    detection_image_path = os.path.join(detection_dir, "images", sample_name + ".jpg")
    detection_label_path = os.path.join(detection_dir, "labels", sample_name + ".txt")
    segmentation_image_dir = os.path.join(segmentation_dir, "image")
    segmentation_mask_png_dir = os.path.join(segmentation_dir, "mask-png")
    segmentation_mask_bmp_dir = os.path.join(segmentation_dir, "mask-bmp")
    segmentation_signal_dir = os.path.join(segmentation_dir, "signal-json")
    temp_bw_path = os.path.join(temp_dir, sample_name + "-bw.jpg")

    # Plot and save detection image
    log = plot_ecg_multilead(
        ecg=signals[sample_idx, :lead_cfg['length'], :lead_cfg['n_leads']].T,
        full_ecg=signals[sample_idx, :, 1].T,
        full_ecg_name=lead_cfg['full_ecg_name'],
        sample_rate=sampling_rate,
        columns=lead_cfg['n_column'],
        lead_index=lead_names,
        title='',
        lead_order=lead_cfg['lead_order'],
        show_lead_name=False,
        show_grid=True,
        show_separate_line=False,
        row_height=row_height,
        style=None,
        save_path=detection_image_path,
        dpi=dpi
    )
    log['padding_x'] = padding_x
    log['padding_y'] = padding_y
    log['border'] = border

    # Save as jpg
    fig = plt.gcf()
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)
    save_figure_as_jpg(detection_image_path, dpi=dpi)

    # Crop and save segmentation image
    crop_lead_images(sample=log, export_path=segmentation_image_dir, img_path=detection_image_path, prefix=sample_name + "_")
    generate_bounding_boxes(sample=log, mode='online', save_bb_path=detection_label_path, img_array=image_array)

    # Generate masks and signal JSONs for each lead
    for lead_num in range(lead_cfg['n_leads'] + 1):
        if lead_num == lead_cfg['n_leads']:
            inserted_signal = np.zeros_like(signals[sample_idx, :lead_cfg['length'], :lead_cfg['n_leads']])
            inserted_full = signals[sample_idx, :, 1]
            if lead_cfg['full_ecg_name'] is None:
                continue
        else:
            inserted_signal = np.zeros_like(signals[sample_idx, :lead_cfg['length'], :lead_cfg['n_leads']])
            inserted_signal[:, lead_num] = signals[sample_idx, :lead_cfg['length'], lead_num]
            inserted_full = np.zeros_like(signals[sample_idx, :, 1])

        plot_ecg_multilead(
            ecg=inserted_signal.T,
            full_ecg=inserted_full.T,
            full_ecg_name=lead_cfg['full_ecg_name'],
            sample_rate=sampling_rate,
            columns=lead_cfg['n_column'],
            lead_index=lead_names,
            title='',
            lead_order=lead_cfg['lead_order'],
            show_lead_name=False,
            show_grid=False,
            show_separate_line=False,
            row_height=row_height,
            style='binary',
            save_path=temp_bw_path,
            dpi=dpi
        )
        fig = plt.gcf()
        fig.canvas.draw()
        save_figure_as_jpg(temp_bw_path, dpi=dpi)

        crop_lead_images(
            sample=log,
            export_path=segmentation_mask_bmp_dir,
            img_path=temp_bw_path,
            prefix=sample_name + "_",
            smaple_number=lead_num,
            save_bmp=True
        )
        crop_lead_images(
            sample=log,
            export_path=segmentation_mask_png_dir,
            img_path=temp_bw_path,
            prefix=sample_name + "_",
            smaple_number=lead_num
        )
        with open(os.path.join(segmentation_signal_dir, f"{sample_name}_lead_{lead_num}.json"), 'w') as f:
            json.dump(log['leads'][lead_num]['ecg'], f)
    return log

# ---------------------- Main Script ----------------------
if __name__ == "__main__":
    load_dotenv()

    args = parse_args()

    # Load configuration from environment or arguments
    SAMPLING_RATE = get_env_int("SAMPLING_RATE", 100)
    DPI = get_env_int("DPI", 700)
    SAMPLE_EACH_LEAD_TRAIN = args.sample_each_lead_train
    SAMPLE_EACH_LEAD_VAL = args.sample_each_lead_val
    SAMPLE_EACH_LEAD_TEST = args.sample_each_lead_test
    PADDING_X = args.padding_x
    PADDING_Y = args.padding_y
    BORDER = args.border
    ROW_HEIGHT = args.row_height

    LEAD_INDEX = get_env_list("LEAD_INDEX", "I,II,III,aVL,aVR,aVF,V1,V2,V3,V4,V5,V6")
    LEAD_DISPLAY = get_env_list("LEAD_DISPLAY", "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6")

    # Lead configuration for different ECG layouts
    LEAD_CONFIG = {
        '3by1': {
            'n_column': get_env_int("LEAD_3BY1_N_COLUMN", 1),
            'length': get_env_int("LEAD_3BY1_LENGTH", 1000),
            'lead_order': list(range(get_env_int("LEAD_3BY1_N_LEADS", 3))),
            'full_ecg_name': None,
            'n_leads': get_env_int("LEAD_3BY1_N_LEADS", 3)
        },
        '3by4': {
            'n_column': get_env_int("LEAD_3BY4_N_COLUMN", 4),
            'length': get_env_int("LEAD_3BY4_LENGTH", 250),
            'lead_order': [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11],
            'full_ecg_name': None,
            'n_leads': get_env_int("LEAD_3BY4_N_LEADS", 12)
        },
    }

    signal_dataset_path = args.signal_dataset_path
    raw_signal_path = args.raw_signal_path
    datasets_path = args.datasets_path

    signal_file = os.path.join(signal_dataset_path, f"all_signals_{SAMPLING_RATE}Hz.npy")

    # Load or generate signal dataset
    if not os.path.exists(signal_file):
        metadata_df = pd.read_csv(os.path.join(raw_signal_path, 'ptbxl_database.csv'), index_col='ecg_id')
        metadata_df.scp_codes = metadata_df.scp_codes.apply(lambda x: ast.literal_eval(x))
        agg_df = pd.read_csv(os.path.join(raw_signal_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        signals = load_raw_data(metadata_df, SAMPLING_RATE, raw_signal_path)
        metadata_df['diagnostic_superclass'] = metadata_df.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
        np.save(signal_file, signals)
        print(f'Signal dataset was saved into {signal_file}')
    else:
        signals = np.load(signal_file)
        print(f'{signal_file} loaded')

    # Prepare output directories
    dataset_root = os.path.join(datasets_path, f"ECG-Dataset")
    detection_dir = os.path.join(dataset_root, "detection")
    segmentation_dir = os.path.join(dataset_root, "segmentation")
    temp_dir = os.path.join(dataset_root, "temp")
    prepare_directories(dataset_root, detection_dir, segmentation_dir, temp_dir)
    print(f'Directory {detection_dir} created.')
    print(f'Directory {segmentation_dir} created.')

    # Get indices for train, val, test splits
    train_indices, end_idx = generate_split_indices(0, LEAD_CONFIG, SAMPLE_EACH_LEAD_TRAIN)
    val_indices, _ = generate_split_indices(end_idx, LEAD_CONFIG, SAMPLE_EACH_LEAD_VAL)
    test_indices, _ = generate_split_indices(signals.shape[0] - SAMPLE_EACH_LEAD_TEST * len(LEAD_CONFIG), LEAD_CONFIG, SAMPLE_EACH_LEAD_TEST)

    logs = {'train': [], 'test': [], 'val': []}

    # Process all samples for each split
    for split_indices, split_name in [(train_indices, "train"), (test_indices, "test"), (val_indices, "val")]:
        for idx, lead_format, lead_cfg in tqdm(split_indices, desc=f"Processing {split_name}"):
            detection_export_path = os.path.join(detection_dir, split_name)
            segmentation_export_path = os.path.join(segmentation_dir, split_name)
            log = process_sample(
                idx, lead_format, lead_cfg, signals, SAMPLING_RATE, ROW_HEIGHT, LEAD_INDEX,
                detection_export_path, segmentation_export_path, temp_dir,
                PADDING_X, PADDING_Y, BORDER, DPI
            )
            logs[split_name].append(log)

    # Save logs for each split
    for split_name, split_logs in logs.items():
        with open(os.path.join(dataset_root, f"logs_segmentation_{split_name}.json"), 'w') as f:
            json.dump({"frequency": SAMPLING_RATE, "samples": split_logs}, f)