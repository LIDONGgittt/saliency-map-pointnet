### This is a re-implement version of "Point Cloud Saliency Maps".
### Introduction
The code is based on https://github.com/charlesq34/pointnet. 

Also, You can find more about "Point Cloud Saliency Maps" on https://github.com/tianzheng4/PointCloud-Saliency-Maps.

The core code of this version are totally different from the https://github.com/tianzheng4/PointCloud-Saliency-Maps. Here are the differences:

- Unlike the original point cloud saliency maps algorithm, which directly deleting high/low saliency score points, this version replace the saliency points' coordinates with the point cloud center's coordinates (normally, 0 for standardized data). 

- In the original algorithm, the ground truth labels are provided to compute the network loss and saliency scores. A new strategy is tried in this version. The algorithm firstly uses the point cloud data to predict the labels. The highest-score labels are extracted to denote the point cloud class. Then the predicted labels are provided to compute the saliency scores. 

The accuracy of ModelNet40 model are shown below:

![ accuracy curves ](https://github.com/LIDONGgittt/saliency-map-pointnet/blob/master/doc/Figure_2.png)

### Usage
The code is based on Python 3.5 and Tensorflow-gpu 1.4.1.

The code needs a well trained PointNet model to compute the saliency maps. Please first refer to https://github.com/charlesq34/pointnet to train a ModelNet40 model. Or, you can download a pre-trained model via https://pan.zju.edu.cn/share/f9bf6bd830f393a091f812800f and extracted it to the base directory.

To compute point cloud saliency maps, use:

`python saliency_map_pointnet`

To see HELP for the script:

`python saliency_map_pointnet -h`

*Contact lidong1994@zju.edu.cn if you have any questions.*
