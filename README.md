## Getting Started
### Setup Environment 
```
conda env create -f environment.yml
conda activate sfda
```
### Test Your Own Samples
#### 1 Config Preparation
- 所有config的默认值可以在 `./conf.py`中修改。
- 有以下两种参数传入方式可以结合使用：
  - 编写`config.yaml`配置文件（在`./cfgs/test_images/`下有用于测试的编写示例），在`demo.sh`的`--cfg`命令参数中传入`config.yaml`的路径。
  - 直接在`demo.sh`的命令中传入（以`demo.sh`中TEST.CKPT_PATH的传入方法为例）
  - ***传参方式优先级：命令行>配置文件>默认值***
- 用于简单测试的配置已经在`demo.sh`中写好，不需更改。
#### 2 Data Preparation
- 在config文件的`DIR:DATASET:`填写`dataset/`目录的路径（**绝对路径**或者**工程目录的相对路径**均可，默认与工程目录并列）
- 测试数据集`test_images/`放在`dataset/`目录下（没有图片的空目录不需要手动添加）。
- 默认的组织结构如下所示。

```
dataset/
└── test_images/
    ├── normal/
    │   └── image/
    │       ├── 000.jpg
    │       ├── 001.jpg
    │       └── ...
    └── anomaly/
        ├── image/
        │   ├── 025.jpg
        │   ├── 026.jpg
        │   └── ...
        └── mask/
            ├── 025.png
            ├── 026.png
            └── ...
ProSFDA/
└── ...
```
#### 3 Launch Demo
```
bash demo.sh
```

#### 4 Get Output
- 测试结果图和日志的输出总目录可由`DIR.OUTPUT`参数指定，默认为`./output/`。