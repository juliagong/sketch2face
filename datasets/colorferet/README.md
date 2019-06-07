# Instructions for using ColorFERET wrangling files

Needs to be at the same scope as a folder called 'colorferet' that contains dvd1 and dvd2 from the raw ColorFERET dataset.

`create_gray_to_color_map.py` creates the pickle map between grayscale and color images in ColorFERET that have sketch files from the CUHK dataset, the names of which are in `sketch_filenames.txt`.
`create_data_from_map.py` then gathers the raw image files by using the pickle map. Before running this, make sure to have directory `data` with subdirectories `color` and `grayscale`.
`randomsplit.py` should be at the same scope as folders `color` and `sketch`, as well as a `test` and `train` folder, each with `color` and `sketch` subdirectories. It splits the data into train and test sets.
`gen_grayscale.py` can be used once another `grayscale` directory is added to both the above train and test directories to generate the corresponding grayscale images for the color ground truth images. 
