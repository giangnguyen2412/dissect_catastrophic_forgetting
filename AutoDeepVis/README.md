# AutoDeepVis

This code implements the method from the paper

"Visualizing Deep Neural Network Decisions: Prediction Difference Analysis" - Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling

which was accepted at ICLR2017, see

https://openreview.net/forum?id=BJ5UeU9xx

### Pytorch implementation
We modified the source code to fit pytorch net and generate IoU comparison for different net maps and ground truth at different layers.

Experiments were done using ResNet50 and testing on MS-COCO dataset.

![](https://github.com/luulinh90s/Explainable-AI-project/tree/master/AutoDeepVis/AutoDeepVis.png)


1. Move your model to "Pytorch_Models" file with the '.ckpt' file to visualized your models.
2. Change the arguement "basenet" to the model you want to compare and "img_shape" to your model input shape in "run_DeepVis.py".
3. If you want to chage the visualized layer, change the "utils_classifier" to the layer you are interested in.
4. Put the figures you want to visualize in folder ./data
5. The output data will be generate at "./results_IoU"
