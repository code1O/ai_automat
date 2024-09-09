#         Essential for tests compiling
# ===================================================
import sys
import os

path_folder = "../../../"

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================