import PySimpleGUI as sg
import os.path
from PIL import Image, ImageTk

# window layout with 2 columns

sg.theme('Dark Blue')

folder_col = [
    [
        sg.Text('Folder'),
        sg.In(size=(25, 1), enable_events=True, key='-FOLDER-'),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20),
            key="-FILE LIST-"
        )
    ],
]

img_frame = [
    [sg.Text('Search Term:'), sg.Input(), sg.Button('Search')],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text('Select Model: '), sg.Combo(['PLACEHOLDER'])],
    [sg.Button('Search Similar'), sg.Button('Exit')]
]

layout = [
    [

        sg.Column(folder_col),
        sg.VerticalSeparator(),
        sg.Column(img_frame),

    ]
]
window = sg.Window("DemoGUI", layout)


def get_file_names(img_paths):
    try:
        file_list = os.listdir(img_paths)
    except:
        file_list = []

    file_names = [
        f
        for f in file_list
        if os.path.isfile(os.path.join(img_paths, f))
        and f.lower().endswith((".png", ".gif", ".jpg"))
    ]
    return file_names


while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == '-FOLDER-':
        fnames = get_file_names(values["-FOLDER-"])
        window["-FILE LIST-"].update(fnames)

    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            img = Image.open(filename)
            window["-IMAGE-"].update(data=ImageTk.PhotoImage(img))

        except:
            pass

window.close()
