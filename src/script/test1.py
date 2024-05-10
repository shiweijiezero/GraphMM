import pickle

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

with open("../../data/traj_data/target_road_lst", "rb") as f:
    trg_road_lst = pickle.load(f)
print(1)