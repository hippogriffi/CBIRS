import PySimpleGUI as sg
import os.path

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import PIL
import operator
import os
from itertools import chain

import global_functions as gf


# window layout with 2 columns

sg.theme('Dark Blue')
global img_db
show = False


# ==================== FOLDER UPLOAD AND SEARCH TAB ==================== #
folder_col = [
    [
        sg.Text('Folder'),
        sg.In(size=(25, 1), enable_events=True, key='folder_upload'),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20),
            key="file_list"
        )
    ],
]

img_frame = [
    [sg.Text('Search Term:', size=(10, 1)), sg.Input(
        size=(15, 1)), sg.Button('Search')],
    [sg.Text(size=(40, 1), key="img_path")],
    [sg.Image(key="img_upload")],
    [sg.Text('Select Model: ', size=(10, 1)), sg.Combo(
        ['M1', 'M3'], size=(15, 1), default_value='  ', enable_events=True, key='model_select')],
    [sg.Button('Search Similar', key='search_btn'),
     sg.Button('DEBUG BTN', key='test_btn'),
     sg.Button('Show Settings', key='settings_toggle'),
     sg.Button('Exit')]
]

M1 = [
    [sg.Frame('Feature Extraction Methods',
              [[sg.Checkbox("Histogram", key='hist_check', default=True),
               sg.Checkbox("Gabor", key='gab_check', default=True),
               sg.Checkbox("Haralick", key='har_check', default=True),
               sg.Checkbox("Dominant Colour", key='dom_check', default=True)]],
              border_width=2)],
    [sg.Frame('Feature Weights',
              [[sg.Slider(range=(0.0, 1.0), key='hist_sdr', orientation='v', size=(6, 64), default_value=0.4, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='gab_sdr', orientation='v', size=(
                   6, 64), default_value=0.1, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='har_sdr', orientation='v', size=(
                   6, 64), default_value=0.1, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='dom_sdr', orientation='v', size=(6, 64), default_value=0.4, resolution=0.1, enable_events=True)]],
              border_width=2)],
]

query_tab = [
    [

        sg.Column(folder_col),

        sg.Frame('', layout=img_frame),
        sg.Column(M1, visible=False,
                  element_justification='l', key='M1'),
    ]
]


# ==================== RETRIVAL RESULTS TAB  ==================== #
retrival_col = [
    [
        sg.Text('Retrival Results'),
        sg.Text('Temp Temp Temp Temp'),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20),
            key="retrival_file_list"
        )
    ],

]

view_retrival_col = [
    [sg.Text(size=(40, 1), key="img_retrival_path")],
    [sg.Image(key="img_retrival")],
    [
        sg.Text('Query Image Distance: '), sg.Text(
            size=(20, 1), key='distance_metric'),
        sg.Button('Histogram')
    ]
]

retrival_tab = [
    [
        sg.Column(retrival_col),
        sg.Column(view_retrival_col),
    ]

]


# ==================== TAB SETUP ==================== #
tab_group_layout = [[
    sg.Tab('Query Select', query_tab, key='query_tab'),
    sg.Tab('Retrival Results', retrival_tab, key='retrival_tab'),
]]

layout = [
    [sg.TabGroup(tab_group_layout, enable_events=True, key='tab_group')],
    [sg.Text('this is a text')]
]


# ==================== SETUP FUNCTIONS ==================== #


# get the surported img files from a folder path
def get_file_names(folder_path):
    full_path_files = []
    root_list = []
    file_names = []

    for (root, dir, file) in os.walk(folder_path, topdown=True):
        root_list.append(root)
        file_names.append(file)

    # remove root dir
    root_list.pop(0)

    for dir in range(len(root_list)):
        for f in file_names[dir+1]:
            temp_name = os.path.join(root_list[dir], f)
            full_path_files.append(temp_name)

    # flatten file names list
    file_names = list(chain.from_iterable(file_names))

    return file_names, full_path_files


# create a list of all imgs in database
def create_img_db(full_fnames):
    global img_db
    img_db = []
    for f in full_fnames:
        img = cv2.imread(f)
        img = cv2.resize(img, (100, 100))
        if img is not None:
            img_db.append(img)

# ==================== SEARCH FUNCTIONS ==================== #


def push_retrival(results):
    retrival_names = []
    full_path = list(map(operator.itemgetter(0), results.items()))
    for f in full_path[:10]:
        retrival_names.append(f.split('/')[-1])
    window["retrival_file_list"].update(retrival_names)


# identifies the model in which the user wants to perform retrival with from combo list
def match_model(model_name, query_img):
    match model_name:
        case 'M1':
            # parameter retrival from gui
            hist_c = values['hist_check']
            gab_c = values['gab_check']
            har_c = values['har_check']
            dom_c = values['dom_check']
            hist_w = values['hist_sdr']
            gab_w = values['gab_sdr']
            har_w = values['har_sdr']
            dom_w = values['dom_sdr']

            results = gf.M1_compute(
                fnames, query_img, img_db, hist_c, gab_c, har_c, dom_c, hist_w, gab_w, har_w, dom_w)
            push_retrival(results)
            return results

        case 'M3':
            results = gf.M3_compute(query_img)

        case _:
            print('Error')

    # ==================== WINDOW SETUP ==================== #
window = sg.Window("CBIR System", layout)
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

# ==================== QUERY WINDOW EVENTS ==================== #
    if event == 'folder_upload':

        fnames, full_fnames = get_file_names(values['folder_upload'])
        window["file_list"].update(full_fnames)
        create_img_db(full_fnames)

    if event == "file_list":
        try:
            filename = os.path.join(
                values["folder_upload"], values["file_list"][0]
            )
            query_img = cv2.imread(filename)
            query_img = cv2.resize(query_img, (100, 100))
            window["img_path"].update(filename)
            viewing_img = Image.open(filename)
            window["img_upload"].update(data=ImageTk.PhotoImage(viewing_img))
        except:
            pass
    if event == 'settings_toggle':
        if (show == False):
            window[values['model_select']].update(visible=True)
            show = True
        else:
            window[values['model_select']].update(visible=False)
            show = False
    if event == 'search_btn':
        retrival_results = match_model(values['model_select'], query_img)

# ==================== RETRIVAL WINDOW EVENTS ==================== #
    if event == 'retrival_file_list':
        try:
            selected_retrival = os.path.join(
                values["folder_upload"], values["retrival_file_list"][0]
            ).replace("\\", "/")
            window["distance_metric"].update(
                retrival_results.get(values["retrival_file_list"][0]))

            window["img_retrival_path"].update(selected_retrival)
            viewing_retrival_img = Image.open(selected_retrival)
            window["img_retrival"].update(
                data=ImageTk.PhotoImage(viewing_retrival_img))
        except:
            pass

# ==================== DEBUG ==================== #
    if event == 'test_btn':
        print(filename)

window.close()
