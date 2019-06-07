import pickle
import shutil
import os

gray_to_color_map = None
with open('gray_to_color_map.pickle', 'rb') as f:
    gray_to_color_map = pickle.load(f)

print(len(gray_to_color_map.keys()))

for gray in gray_to_color_map:
    gray_src = gray
    col_src = gray_to_color_map[gray]
    dst_root = './data/'
    
    gray_name = gray_src.split('/')[-1]
    col_name = col_src.split('/')[-1]

    #print(dst_root + 'grayscale/' + gray_name)
    #print(dst_root + 'color/' + col_name)
    
    #print(gray_src)
    #print(col_src)
    if os.path.isfile(gray_src):
        #pass
        shutil.copy(gray_src, dst_root + 'grayscale/' + gray_name)
    else:
        print(gray_src, col_src)
    if os.path.isfile(col_src):
        #print(col_src)
        #pass
        shutil.copy(col_src, dst_root + 'color/' + col_name)
    else:
        print(gray_src, col_src)
