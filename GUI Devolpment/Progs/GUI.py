import PySimpleGUI as sg
import os.path

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import operator

import global_functions as gf


# window layout with 2 columns

sg.theme('Dark Blue')
global img_db


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
    [sg.Text('Search Term:'), sg.Input(), sg.Button('Search')],
    [sg.Text(size=(40, 1), key="img_path")],
    [sg.Image(key="img_upload")],
    [sg.Text('Select Model: '), sg.Combo(
        ['HIST + Gabor', 'TEMP'], default_value='HIST + Gabor', enable_events=True, key='model_select')],
    [sg.Button('Search Similar', key='search_btn'), sg.Button(
        'DEBUG BTN', key='test_btn'), sg.Button('Exit')]
]

query_tab = [
    [

        sg.Column(folder_col),
        sg.VerticalSeparator(),
        sg.Column(img_frame, key='img_frame'),

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
    file_names = []
    try:
        file_list = os.listdir(folder_path)
    except:
        file_list = []

    file_names = [f for f in file_list if os.path.isfile(os.path.join(
        folder_path, f)) and f.lower().endswith((".png", ".gif", ".jpg"))]

    return file_names

# create a list of all imgs in database


def create_img_db(folder_path, fnames):
    global img_db
    img_db = []
    for f in fnames:
        full_path = os.path.join(folder_path, f)
        img = cv2.imread(full_path)
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

# settings for model 1


def model_1_settings():
    return [
        [sg.Checkbox("Histogram Features: ", key='hist_check')],
        [sg.Checkbox("Gabor Features: ", key='gab_check')],
        [sg.Checkbox("Haralick Features: ", key='har_check')],
        [sg.Checkbox("Dominant Colour Features: ", key='dom_check')],
    ]

# identifies the model in which the user wants to perform retrival with from combo list


def match_model(model_name, query_img):
    match model_name:
        case 'HIST + Gabor':
            results = gf.gabor_hist(fnames, query_img, img_db)
            push_retrival(results)
            return results

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
        fnames = get_file_names(values['folder_upload'])
        window["file_list"].update(fnames)
        create_img_db(values["folder_upload"], fnames)

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
    if event == 'model_select':
        window.extend_layout(window["img_frame"], model_1_settings())
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
        print(retrival_results.get(values["retrival_file_list"][0]))

window.close()
