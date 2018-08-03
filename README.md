# Vieo Action Recognition

This is our project for building the Video Action Recognition.



## Dataset deployment

Dataset deployment steps：

- Make `video-action-recognition/data` directory.

- Download [HMDB51](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar) to the `data` directory– About 2GB for a total of 7,000 clips distributed in 51 action classes. Add use `unrar x xxx.rar` to extract all .rar file in this dataset. Finally, we have an `video-action-recognition/data` with directory tree structure like this: 

  ```shell
  data
  └── HMDB51
      ├── split
      │   ├── README
      │   ├── testTrainMulti_7030_splits
      │   └── test_train_splits.rar
      └── video
          ├── brush_hair
          ├── cartwheel
          ├── catch
          ...
  ```

- Run `dataset/dataset_list_maker.py` to create annotation list file.

  ```
  python dataset/dataset_list_maker.py data/HMDB51/
  ```

- At last,  `video-action-recognition/data` directory tree structure will be like this:

  ```
  data
  └── HMDB51
      ├── meta.txt
      ├── split
      │   ├── README
      │   ├── testTrainMulti_7030_splits
      │   └── test_train_splits.rar
      ├── test_list.txt
      ├── train_list.txt
      └── video
          ├── brush_hair
          ├── cartwheel
          ├── catch
          ...
  ```



## Model training

Resnet_a models:

- Go to the current directory:`cd xxx/video-action-reconigtion` 
- Run tensorboard: `bash tensorboard/tensorboard.sh [port]`. e.g `bash tensorboard/tensorboard.sh 7788`
- Start training with  `bash experiments/scripts/train_resnet_a.sh`



## Experiments result

> We will show the result in validation dataset.
>
> *v1*: In version 1, we use linear classifier to compute every frames scores, and take meas scores as the sample final scores.

| Model                                                     | Frame<br />number | Batch<br />size | Learning<br />rate | Optimizer | Max<br />precision  | Min<br />loss       |
| :-------------------------------------------------------: | :----------: | :--------: | :------------: | :-------: | :-------------------: | :-------------------: |
| ResNet50 + Scores-Average                                 | 10           | 8          | 1e-4      | SGD        | prec=50.07 loss=4.382 | prec=43.33 loss=2.103 |
| ResNet101 + Scores-Average                                | 10           | 8          | 1e-4     | SGD        | prec=50.00 loss=3.032 | prec=43.46 loss=2.039 |
|                                                           |              |            |                |           |                       |                       |
| ResNet50 + Scores-Average                                 | 16           | 8          | 1e-3<br />1e-4 | SGD        | prec=50.85 loss=1.773 | prec=50.85 loss=1.773 |
| ResNet50 + Scores-Average                                 | 32           | 8          | 1e-3<br />1e-4 | SGD        | prec=50.33 loss=3.433 | prec=38.63 loss=2.227 |
| ResNet50 + Scores-Average (Fix-Resnet50)                  | 16           | 8          | 1e-1<br />1e-6 | SGD        | prec=40.78 loss=14.78 | prec=40.00 loss=13.98 |
| ResNet50 + Scores-Average (Fix-Resnet50 except 4th layer) | 16           | 8          | 1e-2<br />1e-6 | SGD        | prec=50.72 loss=3.447 | prec=45.49 loss=3.399 |
| ResNet50 + fc(2048, 1024), fc(1024, 51) + Scores-Average  | 16           | 8          | 1e-3<br />1e-4 | SGD        | prec=46.54 loss=1.957 | prec=46.14 loss=1.953 |
|                                                           |              |            |                |           |                       |                       |
| ResNet18 + Scores-Average                                 | 16           | 8          | 1e-3<br />1e-4 | SGD        | prec=46.41 loss=1.997 | prec=46.41 loss=1.998 |
|                                                           |              |            |                |           |                       |                       |
| ResNet50 + Feature-Average                                |              |            |                |           |                       |                       |
|                                                           |              |            |                |              |           |                  |
| Resnet50 + LSTM（hidden=512 num_layer=1）                 | 16           | 8          | 1e-3<br />1e-4<br />1e-5 | SGD        | prec=49.15 loss=1.816 | prec=48.43 loss=1.795 |
| Resnet50 + LSTM（hidden=1024 num_layer=1）                | 16           | 8          | 1e-3<br />1e-4<br />1e-5 | SGD        | prec=48.50 loss=1.823 | prec=47.78 loss=1.816 |
| Resnet50 + LSTM（hidden=2048 num_layer=1）                | 16           | 8          | 1e-3<br />1e-4<br />1e-5 | SGD        | prec=50.33 loss=1.895 | prec=49.35 loss=1.888 |