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

## Reference
- LAB2-1
  - [pytorch基于resnet18预训练模型用于自己的训练数据集进行迁移学习](https://blog.csdn.net/booklijian/article/details/107214762)
  - [ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)
  - [imgaug数据增强库——学习笔记](https://blog.csdn.net/qq_38451119/article/details/82428612)
- LAB2-2
  - [WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
  - [一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://www.cnblogs.com/marsggbo/p/11308889.html)