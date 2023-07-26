import PySimpleGUI as sg
import os.path

import cv2
from PIL import Image, ImageTk
import operator
import os
from itertools import chain

import global_functions as gf

sg.theme('Dark Blue')
# show settings
SHOW = False
# error bool bcs gui error if trying to access variables that dont exist
ERROR = True

# ==================== FOLDER UPLOAD AND SEARCH TAB ==================== #
# folder upload and file list
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

# image display 
img_frame = [
    [sg.Text('Search Term:', size=(10, 1)), sg.Input(
        size=(15, 1)), sg.Button('Search', enable_events = True, key = 'text_search_btn')],
    [sg.Text(size=(40, 1), key="img_path")],
    [sg.Image(key="img_upload")],
    [sg.Text('Select Model: ', size=(10, 1)), sg.Combo(
        ['M1', 'M2', 'M3'], size=(15, 1), default_value=' ', enable_events=True, key='model_select')],
    [sg.Button('Search Similar', key='search_btn'),
     sg.Button('DEBUG BTN', key='test_btn'),
     sg.Button('Show Settings', key='settings_toggle'),
     sg.Button('Exit')]
]

# model 1 settings
M1 = [
    [sg.Frame('Feature Extraction Methods',
              [[sg.Checkbox("Histogram", key='hist_check_M1', default=True),
               sg.Checkbox("Gabor", key='gab_check_M1', default=True),
               sg.Checkbox("Haralick", key='har_check_M1', default=True),
               sg.Checkbox("Dominant Colour", key='dom_check_M1', default=True)]],
              border_width=2)],
    [sg.Frame('Feature Weights',
              [[sg.Slider(range=(0.0, 1.0), key='hist_sdr_M1', orientation='v', size=(6, 64), default_value=0.4, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='gab_sdr_M1', orientation='v', size=(
                   6, 64), default_value=0.1, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='har_sdr_M1', orientation='v', size=(
                   6, 64), default_value=0.1, resolution=0.1, enable_events=True),
               sg.Slider(range=(0.0, 1.0), key='dom_sdr_M1', orientation='v', size=(6, 64), default_value=0.4, resolution=0.1, enable_events=True)]],
              border_width=2)],
]

# model 2 settings
M2 = [
    [sg.Frame('Feature Extraction Methods',
              [[sg.Checkbox("Colour Histogram", key='hist_check_M2', default=True),
               sg.Checkbox("Gabor", key='gab_check_M2', default=True),
               sg.Checkbox("Haralick", key='har_check_M2', default=True),
               sg.Checkbox("HoG", key='hog_check_M2', default=True)]],
              border_width=2)],
    [sg.Frame('Reigon Based Extraction',
              [[sg.Checkbox("Colour Histogram", key='hist_reg_M2', default=False),
               sg.Checkbox("Gabor", key='gab_reg_M2', default=True),
               sg.Checkbox("Haralick", key='har_reg_M2', default=True),
               sg.Checkbox("HoG", key='hog_reg_M2', default=True)]],
              border_width=2)]
]

# quuery tab layout
query_tab = [
    [

        sg.Column(folder_col),

        sg.Frame('', layout=img_frame),
        sg.Column(M1, visible=False,
                  element_justification='l', key='M1'),
        sg.Column(M2, visible=False,
                  element_justification='l', key='M2'),
        
    ]
]


# ==================== RETRIVAL RESULTS TAB  ==================== #
# retrival results file list
retrival_col = [
    [
        sg.Text('Top Results', size=(25, 1)),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20),
            key="retrival_file_list", no_scrollbar= True
        )
    ],

]

# retrival image display and distance metric
view_retrival_col = [
    [sg.Text(size=(40, 1), key="img_retrival_path")],
    [sg.Image(key="img_retrival")],
    [
        sg.Text('DISTANCE METRIC: '), 
        sg.Text(key='distance_metric')
    ]
]

# retrival tab layout
retrival_tab = [
    [
        sg.Column(retrival_col),
        sg.Column(view_retrival_col, visible= False, key='view_retrival_col')
    ]
]


# ==================== TAB SETUP ==================== #
tab_group_layout = [[
    sg.Tab('Query Select', query_tab, key='query_tab'),
    sg.Tab('Retrival Results', retrival_tab, key='retrival_tab'),
]]

layout = [
    [sg.TabGroup(tab_group_layout, enable_events=True, key='tab_group')],
    [sg.Text('Upload a folder', key = 'sts_msg', text_color= 'yellow')]
]


# ==================== SETUP FUNCTIONS ==================== #


# get the surported img files from a folder path
def get_file_names(folder_path):
    full_path_files, root_list, file_names, multi_dir_names = [],[],[],[]
    DIR_NUM = 0 

    for (root, dir, file) in os.walk(folder_path, topdown=True):
        root_list.append(root)
        # only append surported files types to file list 
        file = [f for f in file if f.lower().endswith((".png", ".jpeg", ".jpg"))]
        file_names.append(file)
        if len(root_list) > 1:
            dir_name = root.rsplit('\\',1)[1]
            file = [dir_name + '\\' + f  for f in file] 
        multi_dir_names.append(file)
        

    #remove root dir if contains other directories 
    if len(root_list) > 1:
        root_list.pop(0)
        multi_dir_names.pop(0)
        DIR_NUM += 1

    for dir in range(len(root_list)):
        for f in file_names[dir+DIR_NUM]:
            temp_name = os.path.join(root_list[dir].replace('/', "\\") , f)
            full_path_files.append(temp_name)

    # flatten file names list
    file_names = list(chain.from_iterable(file_names))
    multi_dir_names = list(chain.from_iterable(multi_dir_names))
    if len(root_list) > 1:
        return multi_dir_names, full_path_files
    else:
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

# push retival results to gui 
def push_retrival(results):
    retrival_names = []
    full_path = list(map(operator.itemgetter(0), results.items()))
    for f in full_path[:20]:
        retrival_names.append(f.split('/')[-1])
    window["retrival_file_list"].update(retrival_names)


# identifies the model in which the user wants to perform retrival with from combo list
def match_model(model_name, query_img):
    
    match model_name:
        case 'M1':
            # parameters from gui
            hist_c = values['hist_check_M1']
            gab_c = values['gab_check_M1']
            har_c = values['har_check_M1']
            dom_c = values['dom_check_M1']
            hist_w = values['hist_sdr_M1']
            gab_w = values['gab_sdr_M1']
            har_w = values['har_sdr_M1']
            dom_w = values['dom_sdr_M1']

            feat_check = [hist_c, gab_c, har_c, dom_c]
            feat_weights = [hist_w, gab_w, har_w, dom_w]
            results = gf.M1_compute(
                fnames, query_img, img_db, feat_check, feat_weights)
            push_retrival(results)
           

        case 'M2':
            # parameters from gui
            hist_c = values['hist_check_M2']
            gab_c = values['gab_check_M2']
            har_c = values['har_check_M2']
            hog_c = values['hog_check_M2']
            hist_r = values['hist_reg_M2']
            gab_r = values['gab_reg_M2']
            har_r = values['har_reg_M2']
            hog_r = values['hog_reg_M2']
            
            feat_check = [hist_c, gab_c, har_c, hog_c]
            reg_check = [hist_r, gab_r, har_r, hog_r]
            results = gf.M2_compute(fnames, query_img, img_db, feat_check, reg_check)
            
        
        case 'M3':
            results = gf.M3_compute(fnames, query_img, img_db)

        case _:
            print('Error')
            return
    
    window['sts_msg'].update('Search successful')
    window['view_retrival_col'].update(visible=True)
    push_retrival(results)
    return results


    # ==================== WINDOW SETUP ==================== #
window = sg.Window("CBIR System", layout)
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

# ==================== QUERY WINDOW EVENTS ==================== #
    # folder upload event
    if event == 'folder_upload':
        # get file names from folder
        fnames, full_fnames = get_file_names(
            values['folder_upload'])
        # update filelist
        window["file_list"].update(fnames)
        # create image database
        create_img_db(full_fnames)
        window['sts_msg'].update(' ')

    # file list event
    if event == "file_list":
        try:
            # get selected file path
            filename = os.path.join(
                values["folder_upload"], values["file_list"][0]
            )
            # get potential query image and resize to model requirements 
            query_img = cv2.imread(filename)
            query_img = cv2.resize(query_img, (100, 100))
            # update filepath in GUI to selected image 
            window["img_path"].update(filename)
            # display image in GUI resized to (300, 300)
            viewing_img = Image.open(filename)
            window["img_upload"].update(data=ImageTk.PhotoImage(viewing_img.resize((300,300))))
        # except clause if file is unsurported
        except:
            pass
    # if I get around to BOW implementation ======
    if event == 'text_search_btn':
        window['sts_msg'].update('BOW search not implemented')
# ==================== MODEL SELECT EVENTS ==================== #
    if event == 'model_select':
        ERROR = False
        if (SHOW == True):
            window['M1'].update(visible=False)
            window['M2'].update(visible=False)
            if values['model_select'] != 'M3':
                window[values['model_select']].update(visible=True)
        
# ==================== SETTINGS TOGGLE EVENTS ==================== #
    if event == 'settings_toggle':
        if (SHOW == False):
            if (ERROR == True):
                window['sts_msg'].update('Select a model before opening model settings')
            else:
                if values['model_select'] == 'M3':
                    window['sts_msg'].update('M3 doesnt have any settings')
                else:
                    window[values['model_select']].update(visible=True)
                    SHOW = True
        else:
            window[values['model_select']].update(visible=False)
            SHOW = False

# ==================== SEARCH BTN EVENTS ==================== #
    if event == 'search_btn': 
        if (ERROR == True):
                window['sts_msg'].update('Select a model before search')
        else:
            retrival_results = match_model(values['model_select'], query_img)

# ==================== RETRIVAL WINDOW EVENTS ==================== #
    if event == 'tab_group':
        window['M1'].update(visible=False)
        window['M2'].update(visible=False)
    if event == 'retrival_file_list':
        try:
            selected_retrival = os.path.join(
                values["folder_upload"], values["retrival_file_list"][0]
            ).replace("\\", "/")
            window["distance_metric"].update(
                retrival_results.get(values["retrival_file_list"][0]))

            window["img_retrival_path"].update(selected_retrival)
            viewing_retrival_img = Image.open(selected_retrival)
            window["img_retrival"].update(data=ImageTk.PhotoImage(viewing_retrival_img.resize((300,300))))
        except:
            pass

# ==================== DEBUG ==================== #
    if event == 'test_btn':
        print("Test")

window.close()
