# Catastrophic forgetting dissector (CFD)

This code implements the method from the paper

"Visualizing Deep Neural Network Decisions: Prediction Difference Analysis" - Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling

which was accepted at ICLR2017, see

https://openreview.net/forum?id=BJ5UeU9xx

Their github code

https://github.com/lmzintgraf/DeepVis-PredDiff
## Pytorch implementation
We modified the source code to fit pytorch net and generate IoU comparison for different net maps and ground truth at different layers.

Experiments were done using ResNet50 and testing on MS-COCO dataset.


## Evaluate customed pytorch models
* Step 1: Move your encoder model files (.ckpt) to `Pytorch_Models/` to visualize your models. 
* Step 2: Change the arguement "basenet" to specify the reference model (the model does not show forgetting) and "img_shape" to your input image shape of your model in "run_CFD.py".

## Tested data
In out experiments, tested data are downloaded from MS-COCO dataset.
* Step 1: Download tested image and ground truth (.npy) for IoU calculatiion from MS-COCO Train/Val annotations and MS-COCO Python API (https://github.com/cocodataset/cocoapi/tree/master/PythonAPI).
To download MS-COCO Train/Val annotations, go to http://cocodataset.org/#download to download.

* Step 2: Move the tested images and ground truth files to `data/`


## Run
Simply run
```
python run_CFD.py
```

## Results
The IoU results and forgetting report will be generated at `IoU_results/`.   
The result format is [test_image]\_[test_model]\_[reference]\_[visualized block].
  
Example:  
`airplane_M19_GT_block1` shows the IoU and feature map between best matching vision of the first block of model "M19" and "Ground Truth" tested on image "airplane".
