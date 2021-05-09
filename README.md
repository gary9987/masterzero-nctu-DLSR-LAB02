masterzero-nctu-DLSR-LAB02
===
## LAB2-1
- After balabced agumentation
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
              iaa.GammaContrast((0.5, 2.0)),
              iaa.Multiply(),
              iaa.GaussianBlur(1.0),
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
## Reference
- LAB2-1
  - [pytorch基于resnet18预训练模型用于自己的训练数据集进行迁移学习](https://blog.csdn.net/booklijian/article/details/107214762)
  - [ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)
  - [imgaug数据增强库——学习笔记](https://blog.csdn.net/qq_38451119/article/details/82428612)
- LAB2-2
  - [WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
  - [一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://www.cnblogs.com/marsggbo/p/11308889.html)