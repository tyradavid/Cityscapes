# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:22:12 2020

@author: tyra1
"""

import os
import glob
import pandas as pd
import json
import pickle

def json_to_csv():
    path_to_json = (r"annotations\train")
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    path_to_png = (r"images\train")
    png_files = [pos_png for pos_png in os.listdir(path_to_png) if pos_png.endswith('.png')]
    fpng=(list(reversed(png_files)))
    n=0
    csv_list = []
    labels=[]
    for j in json_files:
        data_file=open(r"annotations\train\{}".format(j))   
        data = json.load(data_file)
        width,height=data["size"]['width'],data["size"]['height']
        for item in data["objects"]:
            ext = item['points']["exterior"]
            item_class=item["classTitle"]
            xmin=ext[0][0]
            ymin=ext[0][1]
            xmax=ext[1][0]
            ymax=ext[1][1]
            value = (j.split(".")[0] + ".png",
                     width,
                     height,
                     item_class,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            csv_list.append(value)
            n=n+1
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
#    labels_train=list(set(labels))
#    with open("train_labels.txt", "wb") as fp:   #Pickling
#        pickle.dump(labels_train, fp)
    return csv_df

def main():
    csv_df = json_to_csv()
    csv_df.to_csv(r'annotations\train\train_labels.csv', index=None)
    print('Successfully converted json to csv.')

main()