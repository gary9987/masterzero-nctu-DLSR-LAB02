from imgaug import augmenters as iaa
import numpy as np
import imgaug as ia

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply(),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)  #增加飽和度
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import torchvision
    import torchvision.transforms as transforms
    import PIL

    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['figure.figsize'] = 15, 25
    ia.seed(1)

    def show_dataset(dataset, n=6):
        #img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(len(dataset))))
        img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(10)))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


    transform_aug = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(300),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
    ])

    trainset = torchvision.datasets.ImageFolder(root='./food11re/skewed_training',
                                                transform=transform_aug)
    show_dataset(trainset)