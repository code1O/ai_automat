

import os
import sys

#         Essential for compiling
# ===================================================

path_folder = "../"

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))

# ===================================================

from python_utils import handle_json

import ctypes
import subprocess
import time
import tkinter.messagebox as msgbx

def Get_Terminal():
    buffer_length = 1024
    buffer = ctypes.create_unicode_buffer(buffer_length)
    ctypes.windll.kernel32.GetConsoleTitleW(buffer, buffer_length)
    terminal = buffer.value[-8:-4]
    return terminal

this_location, path_project = __file__, __file__[:42]

Log_json = f"{this_location}/Log.json"

terminal_name = Get_Terminal()

clear_shell = lambda: os.system("cls")

# Time for waiting in all functions everytime a command is executed
# measured in seconds
global_time_waiting = 2


if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])