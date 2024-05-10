import argparse
import copy
import io
import math
import os
import pickle
import sys
import tqdm
from PIL import Image
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=2)
parser.add_argument('--end', type=int, default=2)
parser.add_argument('--flag', type=int, default=3)
args = parser.parse_args()

with open("../../data/traj_data/train.src", "rb") as f:
    train_src = pickle.load(f)
with open("../../data/traj_data/train.trg", "rb") as f:
    train_trg = pickle.load(f)
with open("../../data/traj_data/val.src", "rb") as f:
    val_src = pickle.load(f)
with open("../../data/traj_data/val.trg", "rb") as f:
    val_trg = pickle.load(f)
with open("../../data/traj_data/cellID2pos.obj", "rb") as f:
    cellID2pos = pickle.load(f)
with open("../../data/traj_data/roadID2pos.obj", "rb") as f:
    roadID2pos = pickle.load(f)


# 'drive' - 获得可驾驶的公共街道（但不是服务道路）
# 'drive_service' - 获得可驾驶的公共街道，包括服务道路
# 'walk' - 获取行人可以使用的所有街道和路径（这种网络类型忽略单向方向性）
# 'bike' - 获取骑自行车者可以使用的所有街道和路径
# 'all' - 下载所有（非私有）OSM 街道和路径
# 'all_private' – 下载所有 OSM 街道和路径，包括私人访问的
# G = ox.graph_from_place('Hangzhou, China',network_type="drive")
# ox.plot_graph(G)

def transfer_pic(src, trg, mode, start, end):
    # print(src)
    for traj_id in range(len(src)):
        cell_seq = src[traj_id]
        target_seq = trg[traj_id]

        # 将ID转为Pos
        cellpos_seq = [(cellID2pos[i][0], cellID2pos[i][1]) for i in cell_seq]
        targetpos_seq = [roadID2pos[i][0] for i in target_seq]
        targetpos_seq.append(roadID2pos[target_seq[-1]][1])

        # get bottom-left and top-right point
        left = min([i[0] for i in cellpos_seq])
        right = max([i[0] for i in cellpos_seq])
        top = max([i[1] for i in cellpos_seq])
        bottom = min([i[1] for i in cellpos_seq])

        # get big corridor of cellular trajectory
        corridor = LineString(cellpos_seq).buffer(0.015)

        # 只过滤drive
        border_size = 0.002
        # G = ox.graph_from_bbox(top + border_size, bottom - border_size, right + border_size, left - border_size,
        #                        network_type='drive',
        #                        retain_all=False)
        # 使用过滤条件
        # cf='["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]' # tertiary|tertiary_link
        # G = ox.graph_from_bbox(top + border_size, bottom - border_size, right + border_size, left - border_size,
        #                        custom_filter=cf,
        #                        retain_all=False)
        # 得到距离基站序列近的道路
        cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
        G = ox.graph_from_polygon(corridor,
                                  custom_filter=cf,
                                  retain_all=False)

        fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=1,
                                figsize=(2.24, 2.24), bgcolor="#000000", edge_color="#666666")  # 8*8 300dpi
        # 保存路网图
        # fig, ax = ox.plot_figure_ground(G)
        # plt.show()
        # plt.close(fig)
        plt.savefig(f'../../data/pic_store/{mode}/ori/{start + traj_id}.png', format='png')
        #
        # # 添加点，保存src
        x_lst = [i[0] for i in cellpos_seq]
        y_lst = [i[1] for i in cellpos_seq]
        plt.scatter(x_lst, y_lst, c='#ffffff', s=1)  # optional para s=10
        plt.savefig(f'../../data/pic_store/{mode}/src/{start + traj_id}.png', format='png')
        plt.close(fig)
        #
        # # construct target
        # # 添加边，保存trg
        fig2, ax2 = ox.plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=1,
                                  figsize=(2.24, 2.24), bgcolor="#000000", edge_color="#666666")  # 8*8 300dpi
        x_lst = [i[0] for i in targetpos_seq]
        y_lst = [i[1] for i in targetpos_seq]
        plt.plot(x_lst, y_lst, color='#ffffff', linewidth=1)
        plt.savefig(f'../../data/pic_store/{mode}/trg/{start + traj_id}.png', format='png')
        plt.close(fig2)
        print(start + traj_id)


def transfer_pic4tdrive(traj_id,traj_pos_lst,mode="tdrive"):
    # get bottom-left and top-right point
    left = min([i[0] for i in traj_pos_lst])
    right = max([i[0] for i in traj_pos_lst])
    top = max([i[1] for i in traj_pos_lst])
    bottom = min([i[1] for i in traj_pos_lst])

    # get big corridor of cellular trajectory
    corridor = LineString(traj_pos_lst).buffer(0.015)

    # 只过滤drive
    border_size = 0.002
    # 得到距离基站序列近的道路
    cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
    G = ox.graph_from_polygon(corridor,
                              custom_filter=cf,
                              retain_all=False)

    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=1,
                            figsize=(2.24, 2.24), bgcolor="#000000", edge_color="#666666")  # 8*8 300dpi
    # 保存路网图
    # fig, ax = ox.plot_figure_ground(G)
    # plt.show()
    # plt.close(fig)
    plt.savefig(f'../../data/pic_store/{mode}/ori/{traj_id}.png', format='png')
    #
    # # 添加点，保存src
    x_lst = [i[0] for i in traj_pos_lst]
    y_lst = [i[1] for i in traj_pos_lst]
    plt.scatter(x_lst, y_lst, c='#ffffff', s=1)  # optional para s=10
    plt.savefig(f'../../data/pic_store/{mode}/src/{traj_id}.png', format='png')
    plt.close(fig)

if (args.flag == 0):
    transfer_pic(src=train_src[args.start:args.end],
                 trg=train_trg[args.start:args.end],
                 mode="train",
                 start=args.start, end=args.end)
elif (args.flag == 1):
    transfer_pic(src=val_src, trg=val_trg, mode="val", start=args.start, end=args.end)
elif (args.flag == 2):
    # 测试
    transfer_pic(src=train_src[8000:8010],
                 trg=train_trg[8000:8010],
                 mode="train",
                 start=8000, end=8010)
elif (args.flag == 3):
    dir_list=os.listdir("../../data/traj_data/tdrive/")
    for i in tqdm.tqdm(range(len(dir_list))):
        file_name=dir_list[i]
        traj_pos_lst=[]
        dataframe = pd.read_csv(f"../../data/traj_data/tdrive//{file_name}", header=None)

        for index, row in dataframe.iterrows():
            user_id = row[0]
            # time = row[1]
            longitude = row[2]
            latitude = row[3]
            traj_pos_lst.append([longitude,latitude])
        transfer_pic4tdrive(i+1,traj_pos_lst,mode="tdrive")



    # img = Image.open(f'../../data/pic_store/train/src/8001.png')
    # img = img.resize((224, 224))
    # img.save(f'../../data/pic_store/train/src/low_8001.png')
    # img=Image.open(f'../../data/pic_store/{mode}/src/{start + traj_id}.png')
    # img=img.resize((224,224))
    # img.save(f'../../data/pic_store/{mode}/src/low_{start + traj_id}.png')
