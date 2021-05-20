---
tags: Github
---

masterzero-nctu-DLSR-LAB02
===
## LAB2-1
- After balabced agumentation
    ```python=
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
    ```
    ```python=
    wts = [100, 781, 67, 169, 196, 75, 757, 1190, 194, 67, 2857]
    train_dataset.augmentation(wts)
    ```
    ```
    ----------------------------------------------------------------------------------
    Dataset -  ./food11re/skewed_training
    class_name          |bf. loading    |af. loading    
    Bread               |994            |994            
    Dairy_product       |128            |999            
    Dessert             |1500           |1005           
    Egg                 |591            |998            
    Fried_food          |508            |995            
    Meat                |1325           |993            
    Noodles             |132            |999            
    Rice                |84             |999            
    Seafood             |513            |995            
    Soup                |1500           |1005           
    Vegetable_fruit     |35             |999  
    ```
## LAB2-2
- torch.utils.data.RandomSampler
  ```python=
  randon_sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=9000, generator=None)
  ```
  ```
  ----------------------------------------------------------------------------------
  Dataset -  ./food11re/skewed_training
  class_name          |bf. loading    |af. loading    
  Bread               |994            |1253           
  Dairy_product       |128            |167            
  Dessert             |1500           |1843           
  Egg                 |591            |758            
  Fried_food          |508            |606            
  Meat                |1325           |1645           
  Noodles             |132            |143            
  Rice                |84             |102            
  Seafood             |513            |646            
  Soup                |1500           |1789           
  Vegetable_fruit     |35             |48             
  ```
- torch.utils.data.WeightedRandomSampler
  ```python=
  weight = []
  for i in range(11):
      class_count = train_dataset2.targets.count(i)
      weight.append(1./(class_count/len(train_dataset2.targets)))

  samples_weight = np.array([weight[t] for _, t in train_dataset2])
  weighted_sampler = data.WeightedRandomSampler(samples_weight, num_samples=15000, replacement=True)
  ```
  ```
  ----------------------------------------------------------------------------------
  Dataset -  ./food11re/skewed_training
  class_name          |bf. loading    |af. loading    
  Bread               |994            |1323           
  Dairy_product       |128            |1418           
  Dessert             |1500           |1394           
  Egg                 |591            |1359           
  Fried_food          |508            |1366           
  Meat                |1325           |1299           
  Noodles             |132            |1370           
  Rice                |84             |1306           
  Seafood             |513            |1395           
  Soup                |1500           |1373           
  Vegetable_fruit     |35             |1397       
  ```
- torchsampler.ImbalancedDatasetSampler
  ```python=
  train_loader2 = DataLoader(dataset=train_dataset2, num_workers=0, batch_size=100, sampler=ImbalancedDatasetSampler(train_dataset2, num_samples=9000))
  ```
  ```
  ----------------------------------------------------------------------------------
  Dataset -  ./food11re/skewed_training
  class_name          |bf. loading    |af. loading    
  Bread               |994            |806            
  Dairy_product       |128            |831            
  Dessert             |1500           |789            
  Egg                 |591            |801            
  Fried_food          |508            |799            
  Meat                |1325           |849            
  Noodles             |132            |814            
  Rice                |84             |805            
  Seafood             |513            |849            
  Soup                |1500           |826            
  Vegetable_fruit     |35             |831  
  ```
## LAB2-3
- Combine imgaug and torchvision.transform
  ```python=
  class ImgAugTransform:
      def __init__(self):
          self.aug = iaa.SomeOf((1, 2), [
              iaa.GammaContrast((0.5, 2.0)),  # 亮度
              iaa.Multiply(),
              iaa.GaussianBlur(1.0),  # 高斯模糊
              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)  # 增加飽和度
          ])
  
      def __call__(self, img):
          img = np.array(img)
          return self.aug.augment_image(img)
    
  transform_train = transforms.Compose([
          transforms.RandomRotation(90),
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          ImgAugTransform(),
          lambda x: PIL.Image.fromarray(x),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
      ])
  ```
- Using WeightRandomSampler to balance each class. (num_samples=18000)
  ```python=
  weight = []
  for i in range(11):
    class_count = trainset.targets.count(i)
    weight.append(1. / (class_count / len(trainset.targets)))
    
  samples_weight = np.array([weight[t] for _, t in trainset])
  weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, num_samples=18000, replacement=True)
  ```
- Result
  ```
  Top 1 Accuracy of class  0 is 276/368  75.00%
  Top 1 Accuracy of class  1 is  89/148  60.14%
  Top 1 Accuracy of class  2 is 185/231  80.09%
  Top 1 Accuracy of class  3 is 399/500  79.80%
  Top 1 Accuracy of class  4 is 276/335  82.39%
  Top 1 Accuracy of class  5 is 234/287  81.53%
  Top 1 Accuracy of class  6 is 362/432  83.80%
  Top 1 Accuracy of class  7 is 126/147  85.71%
  Top 1 Accuracy of class  8 is  85/ 96  88.54%
  Top 1 Accuracy of class  9 is 262/303  86.47%
  Top 1 Accuracy of class 10 is 463/500  92.60%
  Top 1 accuracy of the network on the 3347 test images: 2757/3347  82.37 %
  82.37227367792053
  
  Top 3 Accuracy of class  0 is 351/368  95.38%
  Top 3 Accuracy of class  1 is 115/148  77.70%
  Top 3 Accuracy of class  2 is 220/231  95.24%
  Top 3 Accuracy of class  3 is 484/500  96.80%
  Top 3 Accuracy of class  4 is 316/335  94.33%
  Top 3 Accuracy of class  5 is 270/287  94.08%
  Top 3 Accuracy of class  6 is 423/432  97.92%
  Top 3 Accuracy of class  7 is 137/147  93.20%
  Top 3 Accuracy of class  8 is  93/ 96  96.88%
  Top 3 Accuracy of class  9 is 284/303  93.73%
  Top 3 Accuracy of class 10 is 492/500  98.40%
  Top 3 accuracy of the network on the 3347 test images: 3185/3347  95.16 %
  95.15984463698835
  ```
- Compare to DLSR-LAB01


    | Class | LAB2 accuracy (%) | LAB1 accuracy (%) |
    | -----:| -----------------:| -----------------:|
    |     0 |             75.00 |             67.93 |
    |     1 |             60.14 |             30.41 |
    |     2 |             80.09 |             54.11 |
    |     3 |             79.80 |             71.40 |
    |     4 |             82.39 |             57.01 |
    |     5 |             81.53 |             70.03 |
    |     6 |             83.80 |             83.10 |
    |     7 |             85.71 |             78.23 |
    |     8 |             88.54 |             79.17 |
    |     9 |             86.47 |             72.28 |
    |    10 |             92.60 |             90.00 |
    |  top1 |             82.37 |             71.35 |
    |  top3 |             95.16 |             91.57 |

## Try to use transfer learning
### Reference
- [pytorch固定部分参数进行网络训练](https://www.jianshu.com/p/fcafcfb3d887)
- [pytorch 固定部分参数训练](https://blog.csdn.net/guotong1988/article/details/79739775)
### Main
- 將包含layer2之前的參數都固定
    ```python=
    net = torchvision.models.resnet18(pretrained=True)
    num_features=net.fc.in_features
    net.fc=nn.Linear(num_features, 11)

    for k,v in net.named_parameters():
        print(k)
        if (k=='conv1.weight' or k == 'bn1.weight' or k == 'bn1.bias'):
            v.requires_grad=False
        if (k[0:6] == 'layer1' or k[0:6] == 'layer2'):
            v.requires_grad=False
    ```
- 檢查各層requires_grad之值
    ```python=
    for k,v in net.named_parameters():
        print(k, v.requires_grad)
    '''
    conv1.weight False
    bn1.weight False
    bn1.bias False
    layer1.0.conv1.weight False
    layer1.0.bn1.weight False
    layer1.0.bn1.bias False
    layer1.0.conv2.weight False
    layer1.0.bn2.weight False
    layer1.0.bn2.bias False
    layer1.1.conv1.weight False
    layer1.1.bn1.weight False
    layer1.1.bn1.bias False
    layer1.1.conv2.weight False
    layer1.1.bn2.weight False
    layer1.1.bn2.bias False
    layer2.0.conv1.weight False
    layer2.0.bn1.weight False
    layer2.0.bn1.bias False
    layer2.0.conv2.weight False
    layer2.0.bn2.weight False
    layer2.0.bn2.bias False
    layer2.0.downsample.0.weight False
    layer2.0.downsample.1.weight False
    layer2.0.downsample.1.bias False
    layer2.1.conv1.weight False
    layer2.1.bn1.weight False
    layer2.1.bn1.bias False
    layer2.1.conv2.weight False
    layer2.1.bn2.weight False
    layer2.1.bn2.bias False
    layer3.0.conv1.weight True
    layer3.0.bn1.weight True
    layer3.0.bn1.bias True
    layer3.0.conv2.weight True
    layer3.0.bn2.weight True
    layer3.0.bn2.bias True
    layer3.0.downsample.0.weight True
    layer3.0.downsample.1.weight True
    layer3.0.downsample.1.bias True
    layer3.1.conv1.weight True
    layer3.1.bn1.weight True
    layer3.1.bn1.bias True
    layer3.1.conv2.weight True
    layer3.1.bn2.weight True
    layer3.1.bn2.bias True
    layer4.0.conv1.weight True
    layer4.0.bn1.weight True
    layer4.0.bn1.bias True
    layer4.0.conv2.weight True
    layer4.0.bn2.weight True
    layer4.0.bn2.bias True
    layer4.0.downsample.0.weight True
    layer4.0.downsample.1.weight True
    layer4.0.downsample.1.bias True
    layer4.1.conv1.weight True
    layer4.1.bn1.weight True
    layer4.1.bn1.bias True
    layer4.1.conv2.weight True
    layer4.1.bn2.weight True
    layer4.1.bn2.bias True
    fc.weight True
    fc.bias True
    '''
    ```
- Optimizer過濾掉不需要訓練的layer (5/20似乎不需要)
    ```python=
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
    ```
- Result
    ```
    Top 1 Accuracy of class  0 is 322/368  87.50%
    Top 1 Accuracy of class  1 is 108/148  72.97%
    Top 1 Accuracy of class  2 is 194/231  83.98%
    Top 1 Accuracy of class  3 is 429/500  85.80%
    Top 1 Accuracy of class  4 is 279/335  83.28%
    Top 1 Accuracy of class  5 is 264/287  91.99%
    Top 1 Accuracy of class  6 is 402/432  93.06%
    Top 1 Accuracy of class  7 is 140/147  95.24%
    Top 1 Accuracy of class  8 is  95/ 96  98.96%
    Top 1 Accuracy of class  9 is 281/303  92.74%
    Top 1 Accuracy of class 10 is 482/500  96.40%
    Top 1 accuracy of the network on the 3347 test images: 2996/3347  89.51 %
    89.51299671347475

    Top 3 Accuracy of class  0 is 365/368  99.18%
    Top 3 Accuracy of class  1 is 134/148  90.54%
    Top 3 Accuracy of class  2 is 224/231  96.97%
    Top 3 Accuracy of class  3 is 490/500  98.00%
    Top 3 Accuracy of class  4 is 318/335  94.93%
    Top 3 Accuracy of class  5 is 284/287  98.95%
    Top 3 Accuracy of class  6 is 427/432  98.84%
    Top 3 Accuracy of class  7 is 144/147  97.96%
    Top 3 Accuracy of class  8 is  96/ 96  100.00%
    Top 3 Accuracy of class  9 is 295/303  97.36%
    Top 3 Accuracy of class 10 is 498/500  99.60%
    Top 3 accuracy of the network on the 3347 test images: 3275/3347  97.85 %
    97.84881983866148
    ```
- Compare to model without pretrained.


    | Class | Transfer learning accuracy(%) | LAB2 accuracy (%) | LAB1 accuracy (%) |
    | -----:| -----------------------------:| -----------------:| -----------------:|
    |     0 |                         87.50 |             75.00 |             67.93 |
    |     1 |                         72.97 |             60.14 |             30.41 |
    |     2 |                         83.98 |             80.09 |             54.11 |
    |     3 |                         85.80 |             79.80 |             71.40 |
    |     4 |                         83.28 |             82.39 |             57.01 |
    |     5 |                         91.99 |             81.53 |             70.03 |
    |     6 |                         93.06 |             83.80 |             83.10 |
    |     7 |                         95.24 |             85.71 |             78.23 |
    |     8 |                         98.96 |             88.54 |             79.17 |
    |     9 |                         92.74 |             86.47 |             72.28 |
    |    10 |                         96.40 |             92.60 |             90.00 |
    |  top1 |                         89.51 |             82.37 |             71.35 |
    |  top3 |                         97.84 |             95.16 |             91.57 |
## Reference
- LAB2-1
  - [pytorch基于resnet18预训练模型用于自己的训练数据集进行迁移学习](https://blog.csdn.net/booklijian/article/details/107214762)
  - [ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)
  - [imgaug数据增强库——学习笔记](https://blog.csdn.net/qq_38451119/article/details/82428612)
- LAB2-2
  - [WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
  - [一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://www.cnblogs.com/marsggbo/p/11308889.html)
  - [WeightedRandomSampler 理解了吧](https://blog.csdn.net/tyfwin/article/details/108435756)