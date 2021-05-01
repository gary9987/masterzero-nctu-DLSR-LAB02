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


## Reference
- [pytorch基于resnet18预训练模型用于自己的训练数据集进行迁移学习](https://blog.csdn.net/booklijian/article/details/107214762)
- [ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)
- [imgaug数据增强库——学习笔记](https://blog.csdn.net/qq_38451119/article/details/82428612)
- [WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)