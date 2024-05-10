import torch
import util
import yaml

# 初始化参数
device = "cpu"
config_dir = 'config.yaml'
with open(config_dir) as fin:
    config = yaml.safe_load(fin)
myutil = util.UtilClass()

# 加载测试数据
with open(f"./data/save_output/src_tensor", "rb") as f:
    src_tensor = torch.load(f, map_location=device)
with open(f"./data/save_output/output_tensor", "rb") as f:
    output_tensor = torch.load(f, map_location=device)
with open(f"./data/save_output/target_tensor", "rb") as f:
    target_tensor = torch.load(f, map_location=device)

# 开始测试
precision, recall = myutil.get_acc(config=config,
                                   src_tensor=src_tensor,
                                   output_tensor=output_tensor,
                                   target_tensor=target_tensor,
                                   batch_idx=0,
                                   batch_size=
                                   # 30,
                                   src_tensor.shape[0],
                                   data_type="val",
                                   save_pic=False)
with open("./validation experiment result.txt", "w") as f:
    f.write(f"precision:{precision}, recall:{recall}")
print(f"precision:{precision}, recall:{recall}")
