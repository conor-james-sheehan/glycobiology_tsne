import os
import requests
from bs4 import BeautifulSoup
from html2text import html2text
import time
import re

from tqdm import tqdm

ROOT_URL = 'http://www.ncbi.nlm.nih.gov'
CONTENTS_URL = ROOT_URL + '/books/NBK310274/'
HTML_LINK_TAG = 'a'
HTML_PAR_TAG = 'p'
HTML_CLASS = 'toc-item'
SAVEDIR = os.path.join('../glycobiology_tsne/data')
CHAPTER_REGEX = re.compile(r'^\d+\. .+$')
CHAPTER_HEADER = """Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology
[Internet]. 3rd edition. Cold Spring Harbor (NY): Cold Spring Harbor
Laboratory Press; 2015-2017. doi: 10.1101/glycobiology.3e.001"""


def _is_chapter(tag):
    try:
        tag_text, = tag.contents
        return CHAPTER_REGEX.match(tag_text)
    except ValueError:
        return False


def _extract_page_contents(url):
    time.sleep(600)
    soup = BeautifulSoup(requests.get(url).content)
    paragraphs = soup.find_all(HTML_PAR_TAG, class_=None)

    def _get_str(tag):
        return html2text(str(tag))

    paragraphs = list(map(_get_str, paragraphs))
    if CHAPTER_HEADER in paragraphs[0]:
        paragraphs = paragraphs[1:]
    txt = ''.join(paragraphs)
    return txt


def _save_chapters(chapters):
    if not os.path.exists(SAVEDIR):
        os.mkdir(SAVEDIR)
    savepath = os.path.join(SAVEDIR, 'textbook.txt')
    with open(savepath, 'w+') as outfile:
        outfile.write(''.join(chapters))


def main():
    contents_webpage = requests.get(CONTENTS_URL)
    contents_soup = BeautifulSoup(contents_webpage.content)
    contents_tags = contents_soup.find_all(HTML_LINK_TAG, HTML_CLASS)
    contents_tags = filter(_is_chapter, contents_tags)
    chapter_links = {tag.contents[0]: ROOT_URL + tag['href'] for tag in contents_tags}
    chapter_contents = {chapter_name: _extract_page_contents(url) for chapter_name, url in tqdm(chapter_links.items())}
    _save_chapters(chapter_contents.values())


if __name__ == '__main__':
    main()
