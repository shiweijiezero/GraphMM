import pickle

with open("../../data/traj_data/baseline_matching_road_lst", "rb") as f:
    matching_road_lst=pickle.load(f)
with open("../../data/traj_data/target_road_lst", "rb") as f:
    trg_road_lst=pickle.load(f)

# 计算精度与召回
all_precision = 0
all_recall = 0
count=0
for i in range(len(matching_road_lst)):
    matching_result=set(matching_road_lst[i])
    target=set(trg_road_lst[i])
    correct_matching = matching_result.intersection(target)
    try:
        precision = len(correct_matching) / len(matching_result)
        recall = len(correct_matching) / len(target)
    except Exception as e:
        print(f"Error:{i},{e},{len(matching_result)}，{len(target)}")
        continue
    print(f"Success:{i},{len(matching_result)}，{len(target)}")
    all_precision += precision
    all_recall += recall
    count += 1
print(f"baseline precision:{all_precision/len(matching_road_lst)}")
print(f"baseline recall:{all_recall/len(matching_road_lst)}")