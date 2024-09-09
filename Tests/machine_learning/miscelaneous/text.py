from compiling_util import SetNavigation

SetNavigation(3)

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from Make_AI import train_text

instance = train_text()
text_pipeline = instance.text_pipeline("Hello World!")
instance.run_model()