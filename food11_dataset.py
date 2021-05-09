
from os.path import basename
import glob
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import PIL
import imgaug as ia
from imgAugTransform import ImgAugTransform
from torchsampler import ImbalancedDatasetSampler


code2names = {
    0:"Bread",
    1:"Dairy_product",
    2:"Dessert",
    3:"Egg",
    4:"Fried_food",
    5:"Meat",
    6:"Noodles",
    7:"Rice",
    8:"Seafood",
    9:"Soup",
    10:"Vegetable_fruit"
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath)
    return img

def input_transform():
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])




class Food11Dataset(data.Dataset):

    def __init__(self, image_dir, input_transform=input_transform, is_train=False):

        super(Food11Dataset, self).__init__()
        self.is_train = is_train
        path_pattern = image_dir + '/**/*.*'
        files_list = glob.glob(path_pattern, recursive=True)
        self.datapath = image_dir
        self.image_filenames = []
        self.num_per_classes = {}
        self.class_start_idx = {}
        for file in files_list:
            if is_image_file(file):
                self.image_filenames.append(file)
                class_name = int(basename(file).split("_")[0])
                if class_name in self.num_per_classes:
                    self.num_per_classes[class_name] += 1
                else:
                    self.num_per_classes[class_name] = 1
                    self.class_start_idx[class_name] = len(self.image_filenames) - 1

        self.input_transform = input_transform

    def __getitem__(self, index):
        # TODO [Lab 2-1] Try to embed third-party augmentation functions into pytorch flow
        input_file = self.image_filenames[index]
        input = load_img(input_file)
        if self.input_transform:
            input = self.input_transform()(input)
        label = basename(self.image_filenames[index])
        label = int(label.split("_")[0])
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def show_details(self):
        for key in sorted(self.num_per_classes.keys()):
            print("{:<8}|{:<20}|{:<12}".format(
                key,
                code2names[key],
                self.num_per_classes[key]
            ))
    
    ''' TODO [Lab 2-1]
    #please add a new function "augmentation(self, wts)"
    #it can change the number of data according to weight of each category
    #"weight" represents the ratio comparing with the amount of original set
    #if the weight > 100, we create new data by copying
    #if the weight < 100, we will delete the original data
    #[hint]you only need to edit the "self.image_filenames" 
    
    wts = [ 125, 80, 25, 100, 200, 800, 80, 60, 40, 150, 1000 ]
    def augmentation(self):
        if is_train:
            pass
    '''
    def augmentation(self, wts):
        if self.is_train:
            image_filenames = []
            classes_num = len(self.num_per_classes)

            for classes in range(classes_num):
                origin = self.num_per_classes[classes]
                ratio = wts[classes]/100
                current_idx = self.class_start_idx[classes]

                if ratio > 1:
                    after = int(origin * ratio)
                    tmp_filenames = self.image_filenames[current_idx:current_idx + origin]
                    count = 0
                    while origin < after:
                        tmp_filenames.append(self.image_filenames[current_idx + count])
                        count += 1
                        count %= self.num_per_classes[classes]
                        origin += 1
                    image_filenames += tmp_filenames
                elif ratio < 1:
                    after = int(origin * ratio)
                    tmp_filenames = self.image_filenames[current_idx:current_idx + after]
                    image_filenames += tmp_filenames
                else:
                    image_filenames += self.image_filenames[current_idx:current_idx+origin]

            self.image_filenames = image_filenames



def data_loading(loader, dataset):

    num_per_classes = {}
    for data_, label in loader:
        for l in label:
            if l.item() in num_per_classes:
                num_per_classes[l.item()] += 1
            else:
                num_per_classes[l.item()] = 1

    print("----------------------------------------------------------------------------------")
    print("Dataset - ", dataset.datapath)
    print("{:<20}|{:<15}|{:<15}".format("class_name", "bf. loading", "af. loading"))
    for key in sorted(num_per_classes.keys()):
        print("{:<20}|{:<15}|{:<15}".format(
            code2names[key],
            dataset.num_per_classes[key],
            num_per_classes[key]
        ))

def main():
    ia.seed(1)

    train_datapath = "./food11re/skewed_training"
    valid_datapath = "./food11re/validation"
    test_datapath = "./food11re/evaluation"

    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = Food11Dataset(train_datapath, is_train=True)
    train_dataset_folder = torchvision.datasets.ImageFolder(root='./food11re/skewed_training', transform=transform)
    valid_dataset = Food11Dataset(valid_datapath, is_train=False)
    test_dataset = Food11Dataset(test_datapath, is_train=False)

    #wts = [100, 781, 67, 169, 196, 75, 757, 1190, 194, 67, 2857]
    #train_dataset.augmentation(wts)


    weight = []
    for i in range(11):
        class_count = train_dataset_folder.targets.count(i)
        weight.append(1./(class_count/len(train_dataset_folder.targets)))

    samples_weight = np.array([weight[t] for _, t in train_dataset_folder])
    weighted_sampler = data.WeightedRandomSampler(samples_weight, num_samples=15000, replacement=True)

    randon_sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=9000, generator=None)


    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", train_datapath)
    print(train_dataset.show_details())

    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", valid_datapath)
    print(valid_dataset.show_details())

    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", test_datapath)
    print(test_dataset.show_details())


    train_folder_loader = DataLoader(dataset=train_dataset_folder, num_workers=0, batch_size=100, sampler=weighted_sampler)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=100, sampler=randon_sampler)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=0, batch_size=100, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=100, shuffle=False)

    data_loading(train_folder_loader, train_dataset)
    data_loading(train_loader, train_dataset)
    data_loading(valid_loader, valid_dataset)
    data_loading(test_loader, test_dataset)


if __name__ == '__main__':
    main()
