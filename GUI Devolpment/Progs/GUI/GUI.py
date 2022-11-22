import PySimpleGUI as sg
import os.path
from PIL import Image, ImageTk

# window layout with 2 columns

sg.theme('Dark Blue')


# folder upload and img viewer tab
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
    [sg.Text('Select Model: '), sg.Combo(['PLACEHOLDER'])],
    [sg.Button('Search Similar'), sg.Button('Exit')]
]

query_tab = [
    [

        sg.Column(folder_col),
        sg.VerticalSeparator(),
        sg.Column(img_frame),

    ]
]

# retrival results tab
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


# tab setup
tab_group_layout = [
    sg.Tab('Query Select', query_tab, key='query_tab'),
    sg.Tab('Retrival Results', retrival_tab, key='retrival_tab')
]

layout = [
    [sg.TabGroup(tab_group_layout, enable_events=True, key='tab_group')]
]

layout = query_tab

window = sg.Window("DemoGUI", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == 'folder_upload':
        folder = values["folder_upload"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg"))
        ]
        window["file_list"].update(fnames)
    if event == "file_list":
        try:
            filename = os.path.join(
                values["folder_upload"], values["file_list"][0]
            )
            window["img_path"].update(filename)
            img = Image.open(filename)
            window["img_upload"].update(data=ImageTk.PhotoImage(img))

        except:
            pass

window.close()
