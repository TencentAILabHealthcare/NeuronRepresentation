import sys
import os

def mkdir(path):
    if os.path.isdir(path):
        return
    os.makedirs(path)
    return

def recur_listdir(path, dir_list):
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            recur_listdir(os.path.join(path, f), dir_list)
        else:
            dir_list.append(os.path.join(path, f))


