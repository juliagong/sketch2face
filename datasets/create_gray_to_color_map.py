import pickle
import re

#g_name = '00001fb010_930831.jpg' # taken from CUHK sketch name file
#g_path = 'feret/cd1/data/pgm/' + g_name[:-3] + 'pgm' # grayscale image path from mapping
#c_path = 'data/images/00001/00001_930831_fb_a.ppm' # color image path from mapping

# grayscale path reformatted to color feret data paths
#new_g_path = './colorferet/dvd2/gray_feret_cd1/data/images/' + g_path[19:-3] + 'tif.bz2'
# color image path reformatted to color feret data paths
#new_c_path = './colorferet/dvd1/' + c_path + '.bz2'

#print(new_g_path, new_c_path)

# get entire gray to color map as dictionary
gray_to_color_map = {}
m = open('./gray_to_color_feret.txt', 'r')
for line in m:
    line = line.split()
    gray_to_color_map[line[0]] = line[1]

# create new gray to color mapping
new_gray_to_color_map = {}
sketch_to_imgs = {} # for mapping sketches to images
f = open('./sketch_filenames.txt')
for line in f:
    g_name = line.strip() # file name from CUHK file list
    cd_number = '1' if int(g_name[:5]) <= 699 else '2'

    # grayscale image path to use in mapping lookup in grayscale-color map
    g_path = 'feret/cd' + cd_number + '/data/pgm/' + g_name[:-3] + 'pgm'
    
    # get color image path from mapping
    if g_path in gray_to_color_map:
        c_path = gray_to_color_map[g_path]
        dvd_number = '1' if int(c_path[12:17]) <= 739 else '2'

        # grayscale path reformatted to color feret data paths
        new_g_path = './colorferet/dvd2/gray_feret_cd' + cd_number + '/data/images/' + g_path[19:-3] + 'tif.bz2'

        # color image path reformatted to color feret data paths
        new_c_path = './colorferet/dvd' + dvd_number + '/' + c_path + '.bz2'

        new_gray_to_color_map[new_g_path] = new_c_path
        sketch_to_imgs[g_name] = [new_g_path, new_c_path]
    
    elif 'd_' not in g_path: # try to salvage the file from fates.txt
        old_g_path = g_path
        # search fates.txt for file
        fates = open('./fates.txt', 'r')
        for line in fates:
            if g_path in line.split()[1]:
                descrip = re.split(' |;', line.strip())
                g_path = [word for word in descrip if '.pgm' in word]
                if len(g_path) <= 1: # there is no correct file
                    g_path = old_g_path
                    continue
                g_path = g_path[1]
                c_path = gray_to_color_map[g_path]
                cd_number = '1' if int(g_path[19:24]) <= 699 else '2'
                dvd_number = '1' if int(c_path[12:17]) <= 739 else '2'

                # grayscale path reformatted to color feret data paths
                new_g_path = './colorferet/dvd2/gray_feret_cd' + cd_number + '/data/images/' + g_path[19:-3] + 'tif.bz2'

                # color image path reformatted to color feret data paths
                new_c_path = './colorferet/dvd' + dvd_number + '/' + c_path + '.bz2'

                new_gray_to_color_map[new_g_path] = new_c_path
                sketch_to_imgs[g_name] = [new_g_path, new_c_path]
                
                break
    else:
        print('did not find file', g_path)

with open('gray_to_color_map.pickle', 'wb') as out:
    pickle.dump(new_gray_to_color_map, out)

with open('sketch_to_img_map.pickle', 'wb') as out:
    pickle.dump(sketch_to_imgs, out)

print('total of', len(new_gray_to_color_map.keys()), 'images processed')
