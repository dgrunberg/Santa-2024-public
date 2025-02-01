import os
import pandas as pd
import hashlib
import io
from sys import getsizeof
from collections.abc import Mapping, Container

def read_words(file, index, results_dir=None):
    #results_dir = f'/mnt/speedy/santa2024/results/{index}'
    #use results_dir if file does not exist and not csv.
    #read the words in file
    #if the file name ends in csv, then read it as csv file and
    #use the location corresponding to index
    dir, tail = os.path.split(file)
    base, ext = os.path.splitext(tail)
    #print(f'{base=}')
    if ext == '.csv':
        df = pd.read_csv(file)
        text = df.loc[index, 'text']
    else:
        #read the file in results and return as a list of words
        for i, prepend in enumerate(['', results_dir]):
            try:
                with open(os.path.join(prepend, file)) as fp:
                    text = fp.read()
                break
            except:
                if i==1:
                    raise ValueError(f'cannot open {file}')
    return text.split()

def read_file(file):
    #read the file in results and return as a list of words
    with open(file) as fp:
        words = fp.read().split(' ')
    return words

def xhash(s, length=6):
    #hash of string s, repeatable
    h=hashlib.new('md5')
    h.update(s.encode())
    return h.hexdigest()[-8:]

sub_data='''
id,text
0,"advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"
1,"advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and"
2,"yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice"
3,"yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice sing cheer and of the is eat visit relax unwrap"
4,"hohoho candle poinsettia snowglobe peppermint eggnog fruitcake chocolate candy puzzle game doll toy workshop wonder believe dream hope peace joy merry season greeting card wrapping paper bow fireplace night cookie milk star wish wreath angel the to of and in that have it not with as you from we kaggle"
5,"advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice sing cheer and of the is eat visit relax unwrap hohoho candle poinsettia snowglobe peppermint eggnog fruitcake chocolate candy puzzle game doll toy workshop wonder believe dream hope peace joy merry season greeting card wrapping paper bow fireplace night cookie milk star wish wreath angel the to of and in that have it not with as you from we kaggle"
'''

def words_translation(index):
    #Gives translation to and from word numbers based on problem index
    df = pd.read_csv(io.StringIO(sub_data))
    words = df.loc[index, 'text'].split()
    word_to_byte={}
    byte_to_word={}
    c=0
    for w in words:
        if w not in word_to_byte:
            word_to_byte[w]=c
            byte_to_word[c]=w
            c+=1
    return word_to_byte, byte_to_word

def num_words(idx):
    #number of words based on problem index
    df = pd.read_csv(io.StringIO(sub_data))
    return len(df.loc[idx, 'text'].split())

def get_batch_size(idx):
    n = num_words(idx)
    batch_size=None
    if n <=30:
        batch_size = 32
    elif n <= 50:
        batch_size = 32
    elif n <= 100:
        batch_size = 25
    return batch_size

import hashlib
def xhash(s, length=6):
    #hash of string s
    h=hashlib.new('md5')
    h.update(s.encode())
    return h.hexdigest()[-8:]

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
This is a recursive function that drills down a Python object graph
like a dictionary holding nested dictionaries with lists of lists
and tuples and sets.
The sys.getsizeof function does a shallow size of only. It counts each
object inside a container as pointer only regardless of how big it
really is.
:param o: the object
:param ids:
:return:
set() is used for ids because it tracks if duplicate objects are pointed to
"""
    d = deep_getsizeof
    if id(o) in ids:
        return 0
    r = getsizeof(o)
    ids.add(id(o))
    if isinstance(o, str) or isinstance(0, str):
        return r
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
    return r 

def memory(x):
    #Simpler version
    r = getsizeof(x)
    if isinstance(x, str):
        return r
    if isinstance(x, Mapping):
        return r + sum(memory(k) + memory(v) for k, v in x.items())
    if isinstance(x, Container):
        return r + sum(memory(y) for y in x)
    return r 
