import PySimpleGUI as sg
import os.path

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
#import Progs.global_functions as gf


# window layout with 2 columns

sg.theme('Dark Blue')
global img_db
global query_img

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
        ['HIST + Gabor', 'TEMP'], default_value='HIST + Gabor', key='model_select')],
    [sg.Button('Search Similar', key='search_btn'), sg.Button(
        'DEBUG BTN', key='test_btn'), sg.Button('Exit')]
]

query_tab = [
    [

        sg.Column(folder_col),
        sg.VerticalSeparator(),
        sg.Column(img_frame),

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


# ==================== FUNCTIONS ==================== #

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


def get_query_img_data(file_path):
    global query_img
    query_img = cv2.imread(file_path)
    query_img = cv2.resize(query_img, (100, 100))

# identifies the model in which the user wants to perform retrival with from combo list


def match_model(model_name):
    match model_name:
        case 'HIST + Gabor':
            # gabor_hist()
            print('hello')

        case _:
            print('Error')


# ==================== WINDOW SETUP ==================== #
window = sg.Window("CBIR System", layout)
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

# ==================== WINDOW EVENTS ==================== #
    if event == 'folder_upload':
        fnames = get_file_names(values['folder_upload'])
        window["file_list"].update(fnames)
        create_img_db(values["folder_upload"], fnames)

    if event == "file_list":
        try:
            filename = os.path.join(
                values["folder_upload"], values["file_list"][0]
            )
            query_img = get_query_img_data(filename)
            window["img_path"].update(filename)
            viewing_img = Image.open(filename)
            window["img_upload"].update(data=ImageTk.PhotoImage(viewing_img))
        except:
            pass
    if event == 'search_btn':
        print(values['model_select'])
        match_model(values['model_select'])

    if event == 'test_btn':
        print(filename)

window.close()
