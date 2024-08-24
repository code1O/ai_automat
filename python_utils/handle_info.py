import pandas
from .datatypes import _Typedata

class handle_json:
    
    def __init__(self, json_content: str, categories: _Typedata, keys: _Typedata) -> None:
        self.json_content = json_content
        self.categories, self.keys = categories, keys
    
    @property
    def get_json(self):
        df = pandas.read_json(self.json_content)
        category_json = (df.get(category) for category in self.categories)
        key_json = (category_json[0][keys] for keys in self.keys)
        return key_json
    
    def update_json(self, new_keys_value):
        keys_to_update = self.get_json
        keys_to_update = (value for value in new_keys_value)
        return keys_to_update