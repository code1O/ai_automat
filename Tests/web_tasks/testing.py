#         Essential for tests compiling
# ===================================================
import sys
import os

path_folder = "../../"

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

from googlesearch import search
from Scripts.handle_web import request_YT

query_search = "Pink Venom"

def google_search(query: str):
    search_query = search(query, num_results=50)
    final_result = ...
    for _, found in enumerate(search_query, 1):
        if ("music.youtube.com" in found) or ("youtu.be" in found):
            final_result = found
            break
            
    return final_result

print(google_search(query_search))