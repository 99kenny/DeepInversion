import os 
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import ast

class DistilledDataset(Dataset):
    def __init__(self, path, transform=None, sub=False):
        super().__init__()
        self.path = path
        self.data = []
        self.transform = transform
        self.class_id = {}
        cls = 0
        print(f'Loading distilled datasets from {path} ...')
        for image in os.listdir(path):
            file = os.path.join(path, image)
            if os.path.isfile(file):
                if sub:
                    real_id = ast.literal_eval(image.split('_')[0])
                    if real_id not in self.class_id:
                        self.class_id[real_id] = cls
                        cls += 1
                    self.data.append([file, self.class_id.get(real_id)])
                else:
                    self.data.append([file, real_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        x = self.transform(Image.open(image))
        y = self.data[index][1]

        return x, y