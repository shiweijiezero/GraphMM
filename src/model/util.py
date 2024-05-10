import os
import pickle
import time

import torch
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from shapely.geometry import LineString, Polygon, MultiPoint, Point
import osmnx as ox
import geopandas
import numpy as np
import torch.nn as nn
import networkx as nx

class UtilClass():
    def __init__(self):
        with open("../../data/traj_data/train.src", "rb") as f:
            self.train_src = pickle.load(f)
        with open("../../data/traj_data/train.trg", "rb") as f:
            self.train_trg = pickle.load(f)
        with open("../../data/traj_data/val.src", "rb") as f:
            self.val_src = pickle.load(f)
        with open("../../data/traj_data/val.trg", "rb") as f:
            self.val_trg = pickle.load(f)
        with open("../../data/traj_data/cellID2pos.obj", "rb") as f:
            self.cellID2pos = pickle.load(f)
        with open("../../data/traj_data/roadID2pos.obj", "rb") as f:
            self.roadID2pos = pickle.load(f)

    def convert_pix2coordinate(self, cell_seq, src_pic, trg_pic):
        """
        将匹配图像像素位置转为对应的经纬坐标
        """
        src_cur = src_pic
        trg_cur = trg_pic

        cellpos_seq = [(self.cellID2pos[i][0], self.cellID2pos[i][1]) for i in cell_seq]

        # 得到 per pix 转换为对应经纬的 距离换算
        top_cell = max([lat for lng, lat in cellpos_seq])
        bottom_cell = min([lat for lng, lat in cellpos_seq])
        left_cell = min([lng for lng, lat in cellpos_seq])
        right_cell = max([lng for lng, lat in cellpos_seq])

        src_idx = torch.argwhere(src_cur == 1)
        down_pic, _ = torch.max(src_idx[:, 0], dim=0)
        right_pic, _ = torch.max(src_idx[:, 1], dim=0)
        top_pic, _ = torch.min(src_idx[:, 0], dim=0)
        left_pic, _ = torch.min(src_idx[:, 1], dim=0)
        # print(down_pic, right_pic, top_pic, left_pic)

        # 换算得到每个像素差对应经纬距离是多少
        height_pix2coordinate = (top_cell - bottom_cell) / (down_pic - top_pic)
        width_pix2coordinate = (right_cell - left_cell) / (right_pic - left_pic)
        # print(height_pix2coordinate,weight_pix2coordinate)

        # 转换匹配结果为一堆 经纬度 point
        res_pos = []
        trg_idx = torch.argwhere(trg_cur == 1)
        for i in range(trg_idx.shape[0]):
            height_loc, width_loc = trg_idx[i][0], trg_idx[i][1]
            # 进行下采样避免后续需要计算的着色点过多
            # if(height_loc%2==0 and width_loc%2==0):
            cur_road_lat = (top_cell - (height_loc - top_pic) * height_pix2coordinate).item()
            cur_road_lon = (left_cell + (width_loc - left_pic) * width_pix2coordinate).item()
            res_pos.append((cur_road_lon, cur_road_lat))

        res_pos = list(set(res_pos))
        return res_pos

    def get_corridor(self, res, radius=0, type="points"):
        if (type == "points"):
            res_corridor = MultiPoint(res).buffer(0.0005)
        else:
            res_corridor = LineString(res).buffer(0.0005)
        if (radius != 0):
            per = 0.001141  # 100m 对应的经度长度
            ex_factor = (per / 100) * radius
            return res_corridor.buffer(ex_factor)
        else:
            return res_corridor

    def get_cmf(self, res, trg, radius=50):
        targetpos_seq = [self.roadID2pos[i][0] for i in trg]
        targetpos_seq.append(self.roadID2pos[trg[-1]][1])
        res_corridor = self.get_corridor(res, radius=radius)

        count = 0
        all_count = len(trg)
        for roadID in trg:
            point_a, point_b = self.roadID2pos[roadID]
            trg_line = LineString([point_a, point_b])
            if (not res_corridor.contains(trg_line)):
                count += 1
        # self.visual_polygon(res_corridor)
        # self.visual_polygon(LineString(targetpos_seq))
        return count / all_count

    def get_rmf(self, res_nodes, trg_nodes):
        correct_nodes = res_nodes.intersection(trg_nodes)
        mismatched_count = len(res_nodes) - len(correct_nodes) + len(trg_nodes) - len(correct_nodes)
        return mismatched_count / len(trg_nodes)

    def get_precision(self, res_nodes, trg_nodes):
        correct_nodes = res_nodes.intersection(trg_nodes)
        return len(correct_nodes) / len(res_nodes)

    def get_recall(self, res_nodes, trg_nodes):
        correct_nodes = res_nodes.intersection(trg_nodes)
        return len(correct_nodes) / len(trg_nodes)

    def get_matched_edges_from_pos(self, G, res_pos):
        X = [item[0] for item in res_pos]
        Y = [item[1] for item in res_pos]
        nearest_edge_lst = ox.nearest_edges(G, X=X, Y=Y, return_dist=False)
        nearest_edge_lst = [item[:2] for item in nearest_edge_lst]
        return nearest_edge_lst

    def visual_pos(self, res_pos):
        x_lst = [i[0] for i in res_pos]
        y_lst = [i[1] for i in res_pos]
        plt.figure(figsize=(4, 4))
        plt.scatter(x_lst, y_lst)
        plt.axis('off')
        plt.show()
        plt.clf()
        plt.close('all')

    def show_pic(self, in_tensor):
        plt.figure(figsize=(4, 4))
        plt.imshow(in_tensor.numpy())
        plt.axis('off')
        plt.show()
        plt.clf()
        plt.close('all')

    def visual_polygon(self, polygon):
        p = geopandas.GeoSeries(polygon)
        p.plot()
        plt.show()
        plt.clf()
        plt.close('all')

    def prepare_tensor(self, src_tensor, output_tensor, target_tensor):
        cell_tensor = torch.where(src_tensor > 0.7, 1, 0)
        road_tensor = torch.where(output_tensor > 0.7, 1, 0)
        target_tensor = torch.where(target_tensor > 0.7, 1, 0)
        return cell_tensor, road_tensor, target_tensor

    def get_acc(self,
                config,
                src_tensor,
                output_tensor,
                target_tensor,
                batch_idx,
                batch_size,
                data_type="val",
                save_pic=False):
        # 基站Figure，匹配路径Figure，ground-truth Figure
        target_tensor = torch.squeeze(target_tensor)
        # 需要对匹配结果在像素级别进行膨胀
        ksize = 2
        max_pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1) / 2))
        output_tensor = max_pool(output_tensor.float())

        cell_tensor, road_tensor, target_road_tensor = self.prepare_tensor(src_tensor, output_tensor, target_tensor)

        # 获得基站位置序列，ground truth道路序列
        with open("../../data/traj_data/target_road_lst", "rb") as f:
            trg = pickle.load(f)
            trg = trg[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if (data_type == "train"):
            cell_src = self.train_src[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # trg = self.train_trg[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        elif (data_type == "val"):
            cell_src = self.val_src[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # trg = self.val_trg[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:
            raise RuntimeError('prepare_acc ERROR! data type is not valid!')

        # 保存可视化图片
        if (save_pic == True):
            [self.save_res_pic(src_tensor[idx], output_tensor[idx], target_tensor[idx], road_tensor[idx],
                               target_road_tensor[idx], batch_idx * batch_size + idx)
             for idx in tqdm.tqdm(range(len(cell_src)))]

        # 为下一次输入准备
        # 基站位置序列，基站Figure，匹配道路Figure，ground-truth道路序列，是否计算
        external_inputs = [(i, cell_src[i], cell_tensor[i], road_tensor[i], trg[i], target_road_tensor[i]) for i in
                           range(len(cell_src))]

        # 路网层面进行匹配，并计算出精度
        # -------------------------
        # 并发计算
        # with torch.multiprocessing.get_context("spawn").Pool(processes=config["multiprocessing"]) as pool:
        #     # print("Get start evaluation!")
        #     acc_lst = pool.map(self.pool_func, external_inputs)
        # -------------------------
        # 循环计算
        acc_lst = [self.pool_func(external_inputs[i]) for i in tqdm.tqdm(range(len(external_inputs))) if
                   len(external_inputs[i][4]) != 0]
        print(f"len(acc_lst):{len(acc_lst)}")
        precision_sum = 0
        recall_sum = 0
        for item in acc_lst:
            precision_sum += item[0]
            recall_sum += item[1]
        return precision_sum / len(cell_src), \
               recall_sum / len(cell_src),

    def pool_func(self, external_input):
        # 基站位置序列，基站Figure，匹配道路Figure，ground-truth道路序列，是否计算
        debug_plot = False
        i, cell_src_i, cell_tensor_i, road_tensor_i, trg_i, target_road_i = external_input
        # 将匹配结果的pixel转换为对应的经纬坐标，输入cell是为了定位比例尺
        start_time = time.time()
        # res_road_pos = self.convert_pix2coordinate(cell_src_i, cell_tensor_i, road_tensor_i)
        # trg_road_pos = self.convert_pix2coordinate(cell_src_i, cell_tensor_i, target_road_i)
        # res_road_pos = torch.tensor(res_road_pos)
        # trg_road_pos = torch.tensor(trg_road_pos)

        # print(f"1-{time.time() - start_time}")
        start_time = time.time()
        # 得到对应的路网图G
        if (os.path.exists(f"./data/save_graphml/graph_{i}.gpkg")):
            G = ox.load_graphml(f"./data/save_graphml/graph_{i}.gpkg")
        else:
            cellpos_seq = [(self.cellID2pos[i][0], self.cellID2pos[i][1]) for i in cell_src_i]
            corridor = LineString(cellpos_seq).buffer(0.015)

            cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
            # G = ox.graph_from_polygon(corridor,
            #                           custom_filter=cf,
            #                           retain_all=False)
            G = ox.graph_from_polygon(corridor,
                                      network_type="drive",
                                      retain_all=False)
            ox.save_graphml(G, filepath=f"./data/save_graphml/graph_{i}.gpkg")
        # print(f"2-{time.time() - start_time}")
        start_time = time.time()

        # 判断地图G中，哪些节点以及哪些道路被着色了
        resulting_roads = self.get_painted_road(G,
                                                cell_src_i,
                                                cell_tensor_i,
                                                road_tensor_i,
                                                debug_plot=debug_plot)
        # res_road_pos = self.convert_pix2coordinate(cell_src_i, cell_tensor_i, road_tensor_i)
        # resulting_roads = self.get_matched_edges_from_pos(G, res_pos=res_road_pos)
        # print(f"3-{time.time() - start_time}")
        start_time = time.time()

        # get metric
        # 首先优先选择着色路段
        new_G = G.copy()
        for edge_id in resulting_roads:
            edge_id = list(edge_id)
            edge_id.append(0)
            new_G.edges[edge_id]['length'] /= 10


        source_node, end_node = trg_i[0][0], trg_i[-1][1]
        if(new_G.nodes.get(source_node)==None):
            source_node = trg_i[1][0]
        if(new_G.nodes.get(end_node)==None):
            end_node = trg_i[-2][1]
        shortest_path_nodes = nx.shortest_path(new_G,source=source_node,target=end_node,
                         weight='length') # 这里返回的是nodes序列
        shortest_path_edges = [
            (shortest_path_nodes[i], shortest_path_nodes[i+1])
            for i in range(len(shortest_path_nodes)-1)] # 变换为edges序列

        resulting_roads = set(shortest_path_edges)
        target_roads = set(trg_i)
        precision = self.get_precision(resulting_roads, target_roads)
        recall = self.get_recall(resulting_roads, target_roads)
        # print(f"6-{time.time() - start_time}")
        # cmf = self.get_cmf(res_road_pos, trg_i, radius=50)
        # rmf = self.get_rmf(res_nodes, trg_nodes)
        print(precision, recall)

        if (debug_plot == True):
            fig, ax = ox.plot_graph(G, show=False, close=False)
            # 绘制最短路径
            ox.plot_graph_route(G, shortest_path_nodes, route_linewidth=6, route_color='green', orig_dest_size=100, ax=ax)
            # 显示图形
            plt.show()
            plt.close()
            try:
                target_path_nodes = []
                for item in trg_i:
                    if(item[0] not in target_path_nodes):
                        target_path_nodes.append(item[0])
                if(trg_i[-1][1] not in target_path_nodes):
                    target_path_nodes.append(trg_i[-1][1])
                fig, ax = ox.plot_graph(G, show=False, close=False)
                ox.plot_graph_route(G, target_path_nodes, route_linewidth=6, route_color='blue', orig_dest_size=100, ax=ax)
                # 显示图形
                plt.show()
                plt.close()
            except:
                pass

        return precision, recall

    def get_painted_road(self, G, cell_seq, src_pic, trg_pic, debug_plot=False):
        cellpos_seq = [(self.cellID2pos[i][0], self.cellID2pos[i][1]) for i in cell_seq]
        # 得到 per pix 转换为对应经纬的 距离换算
        top_cell = max([lat for lng, lat in cellpos_seq])  # 上
        bottom_cell = min([lat for lng, lat in cellpos_seq])  # 下
        left_cell = min([lng for lng, lat in cellpos_seq])  # 左
        right_cell = max([lng for lng, lat in cellpos_seq])  # 右

        src_idx = torch.argwhere(src_pic == 1)
        down_pic, _ = torch.max(src_idx[:, 0], dim=0)  # 图像的下 （左上角为0,0）
        right_pic, _ = torch.max(src_idx[:, 1], dim=0)  # 图像的右
        top_pic, _ = torch.min(src_idx[:, 0], dim=0)  # 图像的上
        left_pic, _ = torch.min(src_idx[:, 1], dim=0)  # 图像的左

        # 换算得到每个经纬差对应多少像素是多少
        height_coordinate2pix = (down_pic - top_pic) / (top_cell - bottom_cell)
        width_coordinate2pix = (right_pic - left_pic) / (right_cell - left_cell)
        # print(height_pix2coordinate,weight_pix2coordinate)

        # 遍历每个道路，判断其是否被着色
        trg_idx = torch.argwhere(trg_pic == 1).numpy().tolist()
        all_painted_road = []
        for G_edge_data in G.edges(keys=False):
            nodes_lst = []
            left_node = (G.nodes[G_edge_data[0]]['x'], G.nodes[G_edge_data[0]]['y'])
            right_node = (G.nodes[G_edge_data[1]]['x'], G.nodes[G_edge_data[1]]['y'])
            # 把一个道路中间的四个位置点放到列表中
            nodes_lst.append(left_node)
            nodes_lst.append(right_node)
            segment_number = 2
            for i in range(1, segment_number):
                nodes_lst.append(
                    (i * abs(left_node[0] - right_node[0]) / segment_number + min(left_node[0], right_node[0]),
                     i * abs(left_node[1] - right_node[1]) / segment_number + min(left_node[1], right_node[1]))
                )
            # 将每个经纬度位置换算成像素位置
            count = 0
            for pos in nodes_lst:
                cur_pic_y = (top_pic + (top_cell - pos[1]) * height_coordinate2pix).item()
                cur_pic_x = (left_pic + (pos[0] - left_cell) * width_coordinate2pix).item()
                cur_pic_y = round(cur_pic_y)
                cur_pic_x = round(cur_pic_x)

                if ([cur_pic_y, cur_pic_x] in trg_idx):
                    count += 1
            if (count >= 2):
                all_painted_road.append(G_edge_data)

        if (debug_plot == True):
            plt.imshow(trg_pic.numpy())
            plt.axis('off')
            ec = ['r' if (G_node_data in all_painted_road)
                  else 'w'
                  for G_node_data in G.edges(keys=False)
                  ]
            fig, ax = ox.plot_graph(G, node_size=0, edge_color=ec, edge_linewidth=0.2)
            plt.close()
        return all_painted_road

    def get_round_pos(self, pos, precision):
        return (round(pos[0], precision), round(pos[1], precision))

    def save_res_pic(self, input_image, output_image, target_image, output_image2, target_image2, idx):
        plt.figure(figsize=(6, 4))

        plt.subplot(2, 2, 0 + 1)
        plt.imshow(input_image.numpy())
        plt.axis('off')

        plt.subplot(2, 2, 1 + 1)
        plt.imshow(output_image.numpy())
        plt.axis('off')

        plt.subplot(2, 2, 2 + 1)
        plt.imshow(target_image.numpy())
        plt.axis('off')

        plt.subplot(2, 2, 3 + 1)
        plt.imshow(output_image2.numpy())
        plt.axis('off')

        plt.subplot(2, 2, 4 + 1)
        plt.imshow(target_image2.numpy())
        plt.axis('off')

        if not os.path.exists("./data/save_picture"):
            os.mkdir("./data/save_picture")

        plt.savefig(
            os.path.join("./data/save_picture",
                         "I{:d}.png".format(idx)),
            dpi=300)
        plt.clf()
        plt.close('all')


# 将图像Tensor转为路网
if __name__ == "__main__":
    myutils = UtilClass()

    output_tensor = torch.load("./out_8", map_location="cpu")
    src_tensor = torch.load("./src_8", map_location="cpu")
    print(output_tensor.shape)
    print(src_tensor.shape)
    # output_tensor = output_tensor[0, 0]
    # src_tensor = src_tensor[0]

    # show_pic(output_tensor)
    # show_pic(src_tensor)

    road_tensor = torch.where(output_tensor > 0.85, 1, 0)
    cell_tensor = torch.where(src_tensor > 0.85, 1, 0)

    # show_pic(road_tensor)
    # show_pic(cell_tensor)

    road_idx = torch.argwhere(road_tensor == 1)
    # print(road_idx)

    batch = 20
    val_src = myutils.val_src[8 * batch]
    val_trg = myutils.val_trg[8 * batch]
    res_pos = myutils.convert_pix2coordinate(val_src, cell_tensor[0], road_tensor[0, 0])

    res_nodes = myutils.get_matched_node_from_pos(val_src, res_pos)
    trg_nodes = myutils.get_nodes_from_trg(val_src, val_trg)

    cmf = myutils.get_cmf(res_pos, val_trg, radius=50)
    rmf = myutils.get_rmf(res_nodes, trg_nodes)
    precision = myutils.get_precision(res_nodes, trg_nodes)
    recall = myutils.get_precision(res_nodes, trg_nodes)
    print(cmf, rmf, precision, recall)
