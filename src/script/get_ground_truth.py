import os
import pickle
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
    filtered_mee_traj =[[(i[0],i[1]) for i in item] for item in list(pickle.load(f).values())]

# # GPS 匹配
trg_road_lst=[]
for i in tqdm.tqdm(range(len(filtered_gps_traj))):
    cell_src = filtered_mee_traj[i]
    gps_src = filtered_gps_traj[i]
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
    gps_src = [(i[1], i[0]) for i in gps_src] # 要求 latitude, longitude
    states, _ = matcher.match(gps_src,tqdm=tqdm.tqdm) # 要求 latitude, longitude
    trg_road_lst.append(states)
    fig, ax = plt.subplots(1, 1)
    mmviz.plot_map(map_con, matcher=matcher,path=gps_src,
                   ax=ax,
                   show_labels=False, show_matching=True, show_graph=True,
                   filename=f"./figure/GPS_{i}.png")
    plt.close()
    # mmviz.plot_map(map_con, path=gps_src, matcher=matcher,
    #                use_osm=True, zoom_path=True,
    #                show_labels=False, show_matching=True, show_graph=False,
    #                filename="my_osm_plot.png")
with open("../../data/traj_data/target_road_lst", "wb") as f:
    pickle.dump(trg_road_lst,f)


# ===================================== #
# 基站匹配
# matching_road_lst=[]
# for i in tqdm.tqdm(range(len(filtered_gps_traj))):
#     cell_src = val_src[i]
#     gps_src = filtered_gps_traj[i]
#     if (os.path.exists(f"../model/data/save_graphml/graph_{i}.gpkg")):
#         G = ox.load_graphml(f"../model/data/save_graphml/graph_{i}.gpkg")
#     else:
#         cellpos_seq = [(cellID2pos[i][0], cellID2pos[i][1]) for i in cell_src]
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
#     matcher = DistanceMatcher(map_con, max_dist=500)
#     cellpos_seq = [(cellID2pos[i][1], cellID2pos[i][0]) for i in cell_src]
#     states, _ = matcher.match(cellpos_seq)
#     matching_road_lst.append(states)
#     fig, ax = plt.subplots(1, 1)
#     mmviz.plot_map(map_con, matcher=matcher,path=gps_src,
#                    ax=ax,
#                    show_labels=False, show_matching=True, show_graph=True,
#                    filename=f"./figure/MEE_{i}.png")
#     plt.close()
# with open("../../data/traj_data/baseline_matching_road_lst", "wb") as f:
#     pickle.dump(matching_road_lst,f)

# ===================================== #
# 计算精度与召回
# with open("../../data/traj_data/baseline_matching_road_lst", "rb") as f:
#     matching_road_lst = pickle.load(f)
# with open("../../data/traj_data/target_road_lst", "rb") as f:
#     trg_road_lst = pickle.load(f)
# all_precision = 0
# all_recall = 0
# count=0
# for i in range(len(matching_road_lst)):
#     matching_result=set(matching_road_lst[i])
#     target=set(trg_road_lst[i])
#     correct_matching = matching_result.intersection(target)
#     try:
#         precision = len(correct_matching) / len(matching_result)
#         recall = len(correct_matching) / len(target)
#     except Exception as e:
#         print(f"Error:{i},{e},{len(matching_result)}，{len(target)}")
#         continue
#     print(f"Success:{i},{len(matching_result)}，{len(target)}")
#     all_precision += precision
#     all_recall += recall
#     count += 1
# print(f"baseline precision:{all_precision/len(matching_road_lst)}")
# print(f"baseline recall:{all_recall/len(matching_road_lst)}")