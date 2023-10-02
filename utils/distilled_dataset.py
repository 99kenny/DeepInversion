import os 
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

class DistilledDataset(Dataset):
    def __init__(self, path, transfrom=None):
        super().__init__()
        self.path = path
        self.data = []
        self.transform = transfrom
        print('Loading distilled datasets from {} ...'.format(path))    
        for image in os.listdir(path):
            file = os.path.join(path, image)
            if os.path.isfile(file):
                class_id = eval(image.split('_')[0])
                self.data.append([file,class_id])
 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index][0]
        x = self.transform(Image.open(image))
        y = self.data[index][1]
        
        return x, y
 