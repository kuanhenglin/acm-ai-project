from ctypes import resize
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

INPUT_WIDTH=200
INPUT_HEIGHT=150

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, i, j, path, device):
        self.path = path
        df = pd.read_csv(path + "/cassava-leaf-disease-classification/train.csv")
        self.pictures = df["image_id"][i:j]
        self.labels = df["label"][i:j]
        self.device = device

        #for i in range(self.pictures.size):
            #if(self.labels[i] != 3):
                #self.pictures.append(self.pictures[i].split()[0] + "_r.jpg")
            
        # if label is 1, 2, 4, 0: add transformations needed to balance data
        # append _something to the end of filename to specify transformation later
        # call .sample to shuffle data afterwards
            

    def __getitem__(self, index):
        # Grab a single training example
        picture = self.pictures.iloc[index]
        # Load and resize the desired image
        im = self.resizeImage(self.path + "/cassava-leaf-disease-classification/train_images/" + picture)
        label = self.labels.iloc[index]
        trans = transforms.ToTensor()
        im = trans(im)
        # perform transformation if filename has _ in it
        
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize(im) # convert image to tensor first!
        example = (im, label)

        return example

    def __len__(self):
        return len(self.pictures)

    def resizeImage(self, im_path):
        im = Image.open(im_path)
        im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
        # im.show()
        return im

# if __name__ == "__main__":
#     datasetInstance = StartingDataset(["data/cassava-leaf-disease-classification/train_images/1000015157.jpg"], [0])
#     datasetInstance.resizeImage(datasetInstance.pictures[0])