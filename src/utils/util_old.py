import pickle
import time

import torch
import matplotlib.pyplot as plt
import tqdm
from shapely.geometry import LineString, Polygon, MultiPoint, Point
import osmnx as ox
import geopandas

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
            cur_cell_lat = top_cell - (height_loc - top_pic) * height_pix2coordinate
            cur_cell_lon = left_cell + (width_loc - left_pic) * width_pix2coordinate
            res_pos.append((cur_cell_lon.item(), cur_cell_lat.item()))

        res_pos = list(set(res_pos))
        return res_pos

    def get_corridor(self, res, radius=0,type="points"):
        if(type=="points"):
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

        count=0
        all_count=len(trg)
        for roadID in trg:
            point_a,point_b=self.roadID2pos[roadID]
            trg_line =  LineString([point_a, point_b])
            if(not res_corridor.contains(trg_line)):
                count+=1
        # self.visual_polygon(res_corridor)
        # self.visual_polygon(LineString(targetpos_seq))
        return count/all_count

    def get_rmf(self,res_nodes,trg_nodes):
        correct_nodes=res_nodes.intersection(trg_nodes)
        mismatched_count=len(res_nodes)-len(correct_nodes) + len(trg_nodes)-len(correct_nodes)
        return mismatched_count/len(trg_nodes)

    def get_precision(self,res_nodes,trg_nodes):
        correct_nodes=res_nodes.intersection(trg_nodes)
        return len(correct_nodes)/len(res_nodes)

    def get_recall(self,res_nodes,trg_nodes):
        correct_nodes=res_nodes.intersection(trg_nodes)
        return len(correct_nodes)/len(trg_nodes)

    def get_matched_node_from_pos(self, cell_seq, res_pos):
        cellpos_seq = [(self.cellID2pos[i][0], self.cellID2pos[i][1]) for i in cell_seq]
        corridor = LineString(cellpos_seq).buffer(0.015)
        cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
        G = ox.graph_from_polygon(corridor,
                                  custom_filter=cf,
                                  retain_all=False)
        nearest_edge_lst = set(
            [ox.nearest_edges(G, X=pos[0], Y=pos[1], interpolate=20, return_dist=False) for pos in res_pos])
        nodes1 = set([node1 for node1, node2, flag in nearest_edge_lst])
        nodes2 = set([node2 for node1, node2, flag in nearest_edge_lst])
        nodes = nodes1.union(nodes2)
        return nodes

    def get_nodes_from_trg(self, cell_seq, trg):
        targetpos_seq = [self.roadID2pos[i][0] for i in trg]
        targetpos_seq.append(self.roadID2pos[trg[-1]][1])
        cellpos_seq = [(self.cellID2pos[i][0], self.cellID2pos[i][1]) for i in cell_seq]
        corridor = LineString(cellpos_seq).buffer(0.015)
        cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
        G = ox.graph_from_polygon(corridor,
                                  custom_filter=cf,
                                  retain_all=False)
        nearest_edge_lst = set(
            [ox.nearest_edges(G, X=pos[0], Y=pos[1], interpolate=20, return_dist=False) for pos in targetpos_seq])
        nodes1 = set([node1 for node1, node2, flag in nearest_edge_lst])
        nodes2 = set([node2 for node1, node2, flag in nearest_edge_lst])
        nodes = nodes1.union(nodes2)
        return nodes

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

    def visual_polygon(self,polygon):
        p = geopandas.GeoSeries(polygon)
        p.plot()
        plt.show()
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

    res_nodes=myutils.get_matched_node_from_pos(val_src, res_pos)
    trg_nodes=myutils.get_nodes_from_trg(val_src, val_trg)

    cmf = myutils.get_cmf(res_pos, val_trg,radius=50)
    rmf = myutils.get_rmf(res_nodes, trg_nodes)
    precision = myutils.get_precision(res_nodes, trg_nodes)
    recall = myutils.get_precision(res_nodes, trg_nodes)
    print(cmf,rmf,precision,recall)
