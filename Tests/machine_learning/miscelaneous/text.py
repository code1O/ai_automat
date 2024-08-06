#         Essential for tests compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from Make_AI import train_text

instance = train_text()
text_pipeline = instance.text_pipeline("Hello World!")
instance.run_model()