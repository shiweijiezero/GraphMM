from data_loader import MyDataset
import torch
from model import MyModel

if __name__ == "__main__":
    my_data = MyDataset(type="train")
    my_model = MyModel()
    input_image_tensor0, trg_image_tensor0 = my_data[0]
    input_image_tensor1, trg_image_tensor1= my_data[1]
    input_image_tensor=torch.stack([input_image_tensor0,input_image_tensor1],dim=0)
    trg_image_tensor=torch.stack([trg_image_tensor0,trg_image_tensor1],dim=0)

    output_image_tensor = my_model(input_image_tensor)
