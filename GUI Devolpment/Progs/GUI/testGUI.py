import PySimpleGUI as sg
import os.path
from PIL import Image, ImageTk

# window layout with 2 columns

sg.theme('Dark Blue')


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


# retrival_tab = [
#     sg.Column(retrival_col),
#     sg.Column(view_retrival_col),
#     sg.HorizontalSeparator(),
#     sg.Column(retrival_stats_col),
# ]


layout = [
    [
        sg.Column(retrival_col),
        sg.Column(view_retrival_col),
    ]

]

window = sg.Window("GUI", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    folder = 'C:/Users/Joe/Desktop/UNI/Yr3/Dissertation/Datasets/101_ObjectCategories/anchor'
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
    window["retrival_file_list"].update(fnames)
    if event == "retrival_file_list":
        try:
            filename = os.path.join(
                folder, values["retrival_file_list"][0]
            )
            img = Image.open(filename)
            window["img_retrival"].update(data=ImageTk.PhotoImage(img))
        except:
            pass

window.close()
