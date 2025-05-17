#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil
import json
from PIL import Image
from dotenv import load_dotenv

# ---------------------- ENVIRONMENT LOADING ----------------------
load_dotenv()

# Load constants from environment or use defaults
text_dict = json.loads(os.getenv("LABELS_DICT", '{"I":1, "II":2, "III":3, "aVR":4, "aVL":5, "aVF":6, "V1":7, "V2":8, "V3":9, "V4":10, "V5":11, "V6":12}'))
lead_index = os.getenv("LEAD_INDEX", "I,II,III,aVL,aVR,aVF,V1,V2,V3,V4,V5,V6").split(",")

# ---------------------- ECG PLOTTING ----------------------
def plot_ecg_multilead(
        ecg,
        full_ecg_name,
        full_ecg,
        sample_rate=500,
        title='ECG 12',
        lead_index=lead_index,
        lead_order=None,
        style=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        save_path: str = None,
        dpi: int = None,
    ):
    """
    Plot multi-lead ECG chart.

    Args:
        ecg: m x n ECG signal data (m: leads, n: signal length)
        full_ecg_name: name of the full ECG (optional)
        full_ecg: signal for the full ECG (optional)
        sample_rate: sampling rate (default 500)
        title: plot title
        lead_index: list of lead names
        lead_order: order of leads to display
        columns: number of columns in plot
        row_height: vertical grid height per lead
        show_lead_name: display lead names
        show_grid: display grid
        show_separate_line: display separating lines
        save_path: if set, save figure to this path
        dpi: figure DPI
        style: plot style ('bw', 'binary', or None)
    Returns:
        output_log: dict with plotting info and bounding boxes
    """
    horizontal_scale = 0.2 #s
    vertical_scale = 0.5 #mv

    fontsize = 6

    num_columns = columns
    num_rows = len(ecg)//num_columns

    xrange_signals = np.arange(ecg.shape[1])/sample_rate/horizontal_scale
    xrange_full = np.arange(full_ecg.shape[0])/sample_rate/horizontal_scale
    range_horizontal = xrange_signals[-1] + 0.2
    # range_vertical = 6

    n_empty_cell_at_left = 3.25   # number of free cells at the least left
    n_empty_cell_at_right = 3 
    n_empty_cell_at_up = 6
    n_empty_cell_at_down = 6

    x_min = -n_empty_cell_at_left
    y_min = -n_empty_cell_at_down

    x_max = range_horizontal*num_columns + n_empty_cell_at_right
    y_max = row_height*(num_rows-1) + n_empty_cell_at_up


    if full_ecg_name:
        y_min -= row_height

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (1, 1, 1)
        face_color = (0, 0, 0)
    if style == 'binary':
        color_major = (0, 0, 0)
        color_minor = (0, 0, 0)
        color_line  = (1, 1, 1)
        face_color = (0, 0, 0)
    else:
        color_major = (0.75, 0.5, 0.5) # red            # changed to have different grid line
        color_minor = (1, 0.93, 0.93)                   # changed
        color_line  = (0,0,0) # black                   # changed
        face_color = (1, 1, 1)

    output_log = {
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "horizontal_scale": horizontal_scale,
        "vertical_scale": vertical_scale,
        "leads": []
    }

    fig, ax = plt.subplots(dpi= dpi)
    for c in range(num_columns):
        for r in range(num_rows):
            idx = c*num_rows + r
            sig = ecg[idx]/vertical_scale
            vertical_starting_point = (num_rows-1-r)*row_height

            is_zero = np.median(sig)==0
            
            x_signal = xrange_signals+c*range_horizontal
            y_signal = sig+vertical_starting_point

            x_text = x_signal[0]
            y_text = vertical_starting_point+2

            bbox = None
            lead_name = None
            if not is_zero:
                ax.plot(x_signal, y_signal, color=color_line, linewidth='0.5')
                ax.set_facecolor(face_color)
                if show_lead_name:
                    bbox = ax.text(x_text,y_text, lead_index[idx], fontsize=fontsize)
                    lead_name = lead_index[idx]

            output_log['leads'].append({
                "min_x_plot": min(x_signal),
                "max_x_plot": max(x_signal),
                "min_y_plot": min(y_signal),
                "max_y_plot": max(y_signal),
                "x_text": x_text,
                "y_text": y_text,
                "bbox": bbox,
                "fontsize": fontsize,
                "text": lead_name,
                "ecg": list(ecg[idx]),
            })

            # add seperating line between leads
            if show_separate_line and not is_zero:
                if c:
                    ax.plot([x_signal[0], x_signal[0]], [y_signal[0] - 0.6, y_signal[0] - 0.2], linewidth='0.5', color=color_line)  # changed
                    ax.plot([x_signal[0], x_signal[0]], [y_signal[0] + 0.2, y_signal[0] + 0.6], linewidth='0.5', color=color_line)  # changed

    if full_ecg_name:
        sig = full_ecg/vertical_scale
        vertical_starting_point = -1*row_height

        is_zero = np.median(sig)==0

        x_signal = xrange_full
        y_signal = sig+vertical_starting_point

        x_text = x_signal[0]
        y_text = vertical_starting_point+2

        bbox = None
        lead_name = None
        if not is_zero:
            ax.plot(x_signal, y_signal, color=color_line, linewidth='0.5')

            if show_lead_name:
                bbox = ax.text(x_text,y_text, full_ecg_name, fontsize=fontsize)
                lead_name_full_ecg = full_ecg_name

        output_log['leads'].append({
            "min_x_plot": min(x_signal),
            "max_x_plot": max(x_signal),
            "min_y_plot": min(y_signal),
            "max_y_plot": max(y_signal),
            "x_text": x_text,
            "y_text": y_text,
            "bbox": bbox,
            "fontsize": fontsize,
            "text": lead_name_full_ecg,
            "ecg": list(full_ecg),
        })

    ax.set_xticks(np.arange(x_min,x_max,1))
    ax.set_yticks(np.arange(y_min,y_max,1))
    # disable the xtickslabels
    ax.set_xticklabels([])                    # changed
    ax.set_yticklabels([])                    # changed

    # disable the frame around the plot
    ax.spines['top'].set_color('none')        # changed
    ax.spines['bottom'].set_color('none')     # changed
    ax.spines['left'].set_color('none')       # changed
    ax.spines['right'].set_color('none')      # changed

    # disable the minor and major ticks
    ax.tick_params(which='both', width=2, color='none')   # changed

    ax.minorticks_on()  
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if show_grid:
        ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color=color_minor)

    ax.set_aspect('equal')
    # plt.axis('off')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout(pad=0)
    if isinstance(save_path, str):
        output_log["save_path"] = save_path
    
    if show_lead_name:
        renderer = fig.canvas.get_renderer()
        for l in output_log['leads']:
            bbox = l['bbox'].get_window_extent(renderer=renderer)
            l['bbox'] = (bbox.x0, bbox.y0, bbox.width, bbox.height)
    

    return output_log   


# ---------------------- FIGURE SAVING ----------------------
def save_figure_as_jpg(save_path, dpi):
    """
    Save the current matplotlib figure as a JPG file.

    Args:
        save_path: output file path
        dpi: dots per inch for output image
    """
    plt.ioff()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

# ---------------------- BOUNDING BOX GENERATION ----------------------
def generate_bounding_boxes(*, sample: dict, mode: str, save_bb_path: str, img_array):
    """
    Generate bounding boxes for ECG leads and optionally save to file.

    Args:
        sample: sample dictionary from logs
        mode: 'load_image' or 'online'
        save_bb_path: output label file path (optional)
        img_array: image array (used in 'online' mode)
    Returns:
        label: list of bounding box tuples or label string
    """
    if mode == 'load_image':
        file_path = sample['save_path']
        file_name = file_path.split("/")[-1].split(".jpg")[0]
        img_array = plt.imread(file_path)
    # elif mode == 'online':
    #     img_array = img_array[:-34, 34:, :]   # we have white border in the left and down of the image, so these borders are removed here
    
    padding_x = sample['padding_x']
    padding_y = sample['padding_y']
    border = sample['border']
    
    img_size = img_array.shape
    # shape of the original image (with white borders)
    image_with_borders_width = img_size[1]
    image_with_borders_height = img_size[0]

    # shape of the target image (without additional borders)
    image_without_borders_width = image_with_borders_width - border
    image_without_borders_height = image_with_borders_height - border

    x_min = sample['x_min']
    x_max = sample['x_max']
    y_min = sample['y_min']
    y_max = sample['y_max']

    width = x_max - x_min
    height = y_max - y_min

    label = []

    for lead in sample['leads']:
        min_x_plot = lead['min_x_plot']
        max_x_plot = lead['max_x_plot']
        min_y_plot = lead['min_y_plot']
        max_y_plot = lead['max_y_plot']


        width_lead = max_x_plot - min_x_plot
        height_lead = max_y_plot - min_y_plot
        
        lead_x_ratio = ((min_x_plot+max_x_plot)/2-x_min)/width
        lead_y_ratio = ((min_y_plot+max_y_plot)/2-y_min)/height

        width_lead_ratio = width_lead/width
        height_lead_ratio = height_lead/height

        # Calculate the coordinates of the box (x, y are centers of the bounding boxes)
        x = int((lead_x_ratio - width_lead_ratio/2) * image_without_borders_width) + border
        y = int((1-(lead_y_ratio + height_lead_ratio/2)) * image_without_borders_height)
        w = int(width_lead_ratio * image_without_borders_width) 
        h = int(height_lead_ratio * image_without_borders_height) 
        
        

        label.append((
            0,
            (x + w/2) / image_with_borders_width,
            (y + h/2) / image_with_borders_height,
            (w + padding_x) / image_with_borders_width,
            (h + padding_y) / image_with_borders_height
        ))

        if lead['text']:        # only if the lead_name is shown
            lead_x_text = lead['x_text']
            lead_y_text = lead['y_text']
            lead_text = lead['text']
            
            text_x0, text_y0, text_width, text_height = lead['bbox']

            x_text_ratio = (lead_x_text-x_min)/width
            y_text_ratio = (lead_y_text-y_min)/height

            label.append((
                text_dict[lead_text],
                (x_text_ratio * image_without_borders_width + border + text_width/2) / image_with_borders_width,
                ((1-y_text_ratio) * image_without_borders_height - text_height/2) / image_with_borders_height,
                (text_width + padding_x) / image_with_borders_width,
                (text_height + padding_y) / image_with_borders_height
            ))
        
    if save_bb_path:
        # change the format of the label in order to write all of the label in one flush rather than writing each line seperately.
        label = '\n'.join(' '.join(str(float_val) for float_val in tup) for tup in label)
        # Write the entire formatted string to the file at once
        with open(save_bb_path, "w") as file:
            file.write(label)

    return label

# ---------------------- DRAW BOUNDING BOXES ----------------------
def draw_bb(sample: dict, export_path: str):
    """
    Draw bounding boxes on ECG image and save.

    Args:
        sample: sample dictionary from logs
        export_path: output image path
    """
    file_path = sample['save_path']
    img = plt.imread(file_path)
    # img = img[:-34, 34:, :]
    img_size = img.shape
    img_width = img_size[1]
    img_height = img_size[0]

    plt.imshow(img)
    for _, x, y, w, h in generate_bounding_boxes(sample=sample, mode='load_image', save_bb_path=None, img_array=None):
        lead_width = w*img_width
        lead_height = h*img_height
        plt.gca().add_patch(plt.Rectangle((x*img_width-lead_width/2, y*img_height-lead_height/2),\
                lead_width, lead_height, edgecolor='g', linewidth=0.3, fill=False))
    plt.show()
    plt.savefig(export_path, dpi=300)
    plt.close()

def crop_lead_images(sample: dict, export_path: str, img_path: str, prefix: str = "", smaple_number: int = None, save_bmp: bool = False):
    """
    Crop individual lead images from an ECG image.

    Args:
        sample: sample dictionary from logs
        export_path: directory to save cropped images
        img_path: path to source ECG image
        prefix: filename prefix for output
        smaple_number: if set, only crop this lead
        save_bmp: if True, save as BMP (binary mask)
    Returns:
        out: list of cropped PIL images
    """
    file_path = sample['save_path']
    img = plt.imread(file_path)
    # img = img[:-34, 34:, :]
    img_size = img.shape
    img_width = img_size[1]
    img_height = img_size[0]

    img = Image.open(img_path)
    lead_number = 0
    out = []

    bbs_info = generate_bounding_boxes(sample=sample, mode='load_image', save_bb_path=None, img_array=None)
    if not isinstance(smaple_number, type(None)):
        if sample['leads'][0]['text']:   # if we have show_lead_name, then this value is true and we have twice bounding boxes (each lead have one bb for text and one for lead)
            bbs_info = [bbs_info[2*smaple_number]]
            lead_number = smaple_number
        else:
            bbs_info = [bbs_info[smaple_number]]
            lead_number = smaple_number
    
    for c, x, y, w, h in bbs_info:
        if c != 0:
            continue
        lead_width = w*img_width
        lead_height = h*img_height
        start_x, start_y = x*img_width-lead_width/2, y*img_height-lead_height/2
        end_x, end_y = x*img_width+lead_width/2, y*img_height+lead_height/2
        cropped = img.crop((start_x, start_y, end_x, end_y))
        out.append(cropped)
        if export_path:
            if save_bmp:
                cropped = np.array(cropped)
                cropped = cropped>100
                cropped = np.uint8(cropped)
                plt.imsave(os.path.join(export_path, f"{prefix}lead_{lead_number}.bmp"), cropped, cmap='gray', format='bmp')
            else:
                cropped.save(os.path.join(export_path, f"{prefix}lead_{lead_number}.png"))
        lead_number += 1
    return out


def generate_split_indices(start_index, lead_config, number_each_lead):
    """
    Generate indices for dataset splits.

    Args:
        start_index: starting index
        lead_config: dict of lead configurations
        number_each_lead: number of samples per lead
    Returns:
        out: list of (index, lead_format, config) tuples
        idx: next available index after split
    """
    out = []
    idx = start_index
    for i in range(number_each_lead):
        for lead_format, each_lead_config in lead_config.items():
            out.append((idx, lead_format, each_lead_config))
            idx += 1
    return out, idx

# ---------------------- MAIN (TESTING) ----------------------
if __name__ == "__main__":
    # Example usage: draw bounding boxes on first 4 samples from logs
    log_path = "datasets/image_dataset_v4.0/detection/train/"
    with open(f"{log_path}logs.json", 'r') as f:
        logs = json.load(f)
    for i in range(4):
        draw_bb(logs['samples'][i], f"{i}.jpg")

