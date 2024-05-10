import json
import os
import pickle
from multiprocessing import Pool

import tqdm
from shapely.geometry import LineString, Polygon, MultiPoint, Point
import osmnx as ox
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import matplotlib.pyplot as plt

# 初始化参数
device = "cpu"

# 加载测试数据
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

with open("../../data/traj_data/filtered_gps_traj.obj", "rb") as f:
    filtered_gps_traj = [[(i[0], i[1]) for i in item] for item in list(pickle.load(f).values())]
with open("../../data/traj_data/filtered_mee_traj.obj", "rb") as f:
    filtered_mee_traj = [[(i[0], i[1]) for i in item] for item in list(pickle.load(f).values())]


def get_GPS_res(params):
    i, cell_src, gps_src = params
    if (os.path.exists(f"../model/data/save_graphml/graph_{i}.gpkg")):
        G = ox.load_graphml(f"../model/data/save_graphml/graph_{i}.gpkg")
    else:
        cellpos_seq = cell_src
        corridor = LineString(cellpos_seq).buffer(0.015)

        cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
        # G = ox.graph_from_polygon(corridor,
        #                           custom_filter=cf,
        #                           retain_all=False)
        G = ox.graph_from_polygon(corridor,
                                  network_type="drive",
                                  retain_all=False)
        ox.save_graphml(G, filepath=f"../model/data/save_graphml/graph_{i}.gpkg")

    map_con = InMemMap("myosm", index_edges=True)
    # graph_proj = ox.project_graph(G, to_crs=4326)

    # Create GeoDataFrames (gdfs)
    # Approach 1
    nodes_proj, edges_proj = ox.graph_to_gdfs(G, nodes=True, edges=True)
    for nid, row in nodes_proj[['x', 'y']].iterrows():
        map_con.add_node(nid, (row['y'], row['x']))
    for eid, _ in edges_proj.iterrows():
        map_con.add_edge(eid[0], eid[1])

    matcher = DistanceMatcher(map_con, max_dist=300)
    gps_src = [(i[1], i[0]) for i in gps_src]  # 要求 latitude, longitude
    states, _ = matcher.match(gps_src, tqdm=tqdm.tqdm)  # 要求 latitude, longitude
    fig, ax = plt.subplots(1, 1)
    mmviz.plot_map(map_con, matcher=matcher, path=gps_src,
                   ax=ax,
                   show_labels=False, show_matching=True, show_graph=True,
                   filename=f"./figure/GPS_{i}.png")
    plt.close()
    return states


if __name__ == "__main__":
    pool = Pool(64)
    # Run
    trg_road_lst = pool.map(
        get_GPS_res,
        [
            (i, filtered_mee_traj[i], filtered_gps_traj[i])
            for i in range(len(filtered_gps_traj))
        ]
    )
    pool.close()
    pool.join()
    # json格式保存
    with open("../../data/traj_data/target_road_lst.json", "wb") as f:
        json.dump(trg_road_lst, f, ensure_ascii=False, indent=4)

    #
    #
    # trg_road_lst = []
    # for i in tqdm.tqdm(range(len(filtered_gps_traj))):
    #     cell_src = filtered_mee_traj[i]
    #     gps_src = filtered_gps_traj[i]
    #     if (os.path.exists(f"../model/data/save_graphml/graph_{i}.gpkg")):
    #         G = ox.load_graphml(f"../model/data/save_graphml/graph_{i}.gpkg")
    #     else:
    #         cellpos_seq = cell_src
    #         corridor = LineString(cellpos_seq).buffer(0.015)
    #
    #         cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'  # tertiary|tertiary_link
    #         # G = ox.graph_from_polygon(corridor,
    #         #                           custom_filter=cf,
    #         #                           retain_all=False)
    #         G = ox.graph_from_polygon(corridor,
    #                                   network_type="drive",
    #                                   retain_all=False)
    #         ox.save_graphml(G, filepath=f"../model/data/save_graphml/graph_{i}.gpkg")
    #
    #     map_con = InMemMap("myosm", index_edges=True)
    #     # graph_proj = ox.project_graph(G, to_crs=4326)
    #
    #     # Create GeoDataFrames (gdfs)
    #     # Approach 1
    #     nodes_proj, edges_proj = ox.graph_to_gdfs(G, nodes=True, edges=True)
    #     for nid, row in nodes_proj[['x', 'y']].iterrows():
    #         map_con.add_node(nid, (row['y'], row['x']))
    #     for eid, _ in edges_proj.iterrows():
    #         map_con.add_edge(eid[0], eid[1])
    #
    #     matcher = DistanceMatcher(map_con, max_dist=300)
    #     gps_src = [(i[1], i[0]) for i in gps_src]  # 要求 latitude, longitude
    #     states, _ = matcher.match(gps_src, tqdm=tqdm.tqdm)  # 要求 latitude, longitude
    #     trg_road_lst.append(states)
    #     fig, ax = plt.subplots(1, 1)
    #     mmviz.plot_map(map_con, matcher=matcher, path=gps_src,
    #                    ax=ax,
    #                    show_labels=False, show_matching=True, show_graph=True,
    #                    filename=f"./figure/GPS_{i}.png")
    #     plt.close()
    #     # mmviz.plot_map(map_con, path=gps_src, matcher=matcher,
    #     #                use_osm=True, zoom_path=True,
    #     #                show_labels=False, show_matching=True, show_graph=False,
    #     #                filename="my_osm_plot.png")
    # with open("../../data/traj_data/target_road_lst", "wb") as f:
    #     pickle.dump(trg_road_lst, f)
