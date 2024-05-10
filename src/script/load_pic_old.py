import io
import os
import pickle
import sys
import tqdm
import folium
from PIL import Image
from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj

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


def transfer_pic(src, trg, mode):
    for traj_id in tqdm.tqdm(range(len(src))):
        cell_seq = src[traj_id]
        target_seq = trg[traj_id]

        # 将ID转为Pos
        # cellpos_seq = [(cellID2pos[i][0], cellID2pos[i][1]) for i in cell_seq]
        # targetpos_seq = [roadID2pos[i][0] for i in target_seq]
        # targetpos_seq.append(roadID2pos[target_seq[-1]][1])
        # 高德坐标系需要GCJ-02
        cellpos_seq = [wgs2gcj(cellID2pos[i][0], cellID2pos[i][1]) for i in cell_seq]
        targetpos_seq = [wgs2gcj(*roadID2pos[i][0]) for i in target_seq]
        targetpos_seq.append(wgs2gcj(*roadID2pos[target_seq[-1]][1]))



        # 将(经度，纬度)变为(纬度，经度)
        cellpos_seq = [(i[1], i[0]) for i in cellpos_seq]
        targetpos_seq = [(i[1], i[0]) for i in targetpos_seq]

        # get bottom-left and top-right point
        left = min([i[1] for i in cellpos_seq])
        right = max([i[1] for i in cellpos_seq])
        top = max([i[0] for i in cellpos_seq])
        bottom = min([i[0] for i in cellpos_seq])

        # initiate to plotting area
        my_map = folium.Map(location=[cellpos_seq[0][0], cellpos_seq[-1][1]],
                            # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                            # 高德街道图
                            # tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图
                            # tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', # google 卫星图
                            # tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', # google 地图
                            #
                            tiles="http://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                            attr='default',
                            png_enabled=True
                            )
        # loop each point
        folium.PolyLine(cellpos_seq, color="red", weight=5, opacity=0.4).add_to(my_map)
        # initiate to plotting area
        my_map_trg = folium.Map(location=[cellpos_seq[0][0], cellpos_seq[-1][1]],
                                # zoom_start=15,
                                # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                                # 高德街道图
                                # tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图
                                # tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', # google 卫星图
                                # tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', # google 地图
                                #
                                tiles="http://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                                attr='default',
                                # zoom_control=False,
                                # scrollWheelZoom=False,
                                # dragging=False,
                                png_enabled=True)
        folium.PolyLine(targetpos_seq, color="green", weight=7, opacity=1).add_to(my_map_trg)

        sw = [bottom, left]
        ne = [top, right]
        # print([sw, ne])
        my_map.fit_bounds([sw, ne])
        my_map_trg.fit_bounds([sw, ne])

        img_data = my_map._to_png(1)
        img = Image.open(io.BytesIO(img_data))
        img.save(f"../../data/pic_store/{mode}/src/{traj_id}.png")
        img_data = my_map_trg._to_png(3)
        img = Image.open(io.BytesIO(img_data))
        img.save(f"../../data/pic_store/{mode}/trg/{traj_id}.png")

        my_map.save(f"../../data/pic_store/{mode}/src/{traj_id}.html")
        my_map_trg.save(f"../../data/pic_store/{mode}/trg/{traj_id}.html")


transfer_pic(src=train_src[0:2], trg=train_trg[0:2], mode="train")
# transfer_pic(src=val_src,trg=val_trg,mode="test")
