import re
import os.path as path
import urllib
import urllib.error
import urllib.request
import requests
from googletrans import Translator
from pytube import YouTube
from bs4 import BeautifulSoup as Bs4
from googlesearch import search

re_chars = re.compile("<.*?>")

def check_connection():
    url = "https://google.com"
    try:
        urllib.request.urlopen(url)
        return True
    except urllib.error.URLError:
        return False

def get_lyrics(author, song) -> str:
    link = f"letras.com/{author}/{song}/"
    link_requests = requests.get(link)
    html = Bs4(link_requests.content, "html.parser")
    soup = html.find_all("div", {"class": "lyric-original font --lyrics --size18"})
    cleaned_result = re.sub(soup, "", re_chars)
    return cleaned_result

def translate(text, destiny):
    translator = Translator()
    translation_text = translator.translate(text, destiny)
    return translation_text

class extract_query:
    """
    Extract information from trusted websites and resources
    
    For Academic purposes
    - Redalyc
    - Google Academic
    
    """
    from pypdf import PdfReader
    from io import BytesIO
    
    def pdf_reader(self, content, number_page: int=0):
        pdf_file = self.BytesIO(content)
        reader = self.PdfReader(pdf_file)
        metadata = reader.metadata
        page = reader.get_page(page_number=number_page)
        text = page.extract_text()
        dictionary = dict(pdf_page=page, pdf_text=text, pdf_metadata=metadata)
        return dictionary
    
    def __init__(self, query: str) -> None:
        self.query_normal = query
        self.query_cleaned = query.replace(" ", "%20")
    
    def search_academic(self) -> str:
        link_host = "https://redalyc.org"
        key = "busquedaArticuloFiltros.oa?q="
        fetch_host = requests.get(f"{link_host}/{key}/{self.query_cleaned}")
        html = Bs4(fetch_host.content, "html.parser")
        content = html.find(class_="wrapper")
        href = content.find_all("a", {"class":""}, href=True)
        
class request_YT(YouTube):
    def __init__(self, query, path_destiny, file_name) -> None:
        super().__init__(query)
        
        if path.exists(path.join(path_destiny, file_name)):
            self.path, self.file_name = 0, 0
        
        self.path, self.file_name = path_destiny, file_name
        self.directory = path.join(path_destiny, file_name)
        
    @property
    def download_as_mp4(self) -> str:
        streams = self.streams.get_highest_resolution()
        
        if (self.file_name != 0) or (self.path != 0):
            streams.download(self.path, self.file_name)
            
        return f"{self.directory} already exists!, try again with other directory"
    
    @property
    def download_as_mp3(self) -> str:
        streams = self.streams.get_audio_only(subtype="mp3")
        
        if (self.file_name != 0) or (self.path != 0):
            streams.download(self.path, self.file_name)
        
        return f"{self.directory} already exists! try again with other directory"
    
    @property
    def get_items(self):
        
        author, name_video = self.author, self.title
        embed_html = self.embed_html
        
        dictionary = dict(embed_html=embed_html, author=author, name_video=name_video)
        
        return dictionary
