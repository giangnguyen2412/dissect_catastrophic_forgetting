# Catastrophic forgetting dissector (CFD)
We adopt PDA from Zintgraf from [HERE](https://github.com/lmzintgraf/DeepVis-PredDiff)
## Pytorch implementation
We modified the source code of PDA from Caffe to Pytorch to fit networks and generate IoU comparison for different net maps and ground truth at different layers.

Experiments were done using ResNet50 and testing on MS-COCO dataset.

## Evaluate customized pytorch models
After training, you will get `encoder.ckpt` and `decoder.cpkt`, we take `encoder.ckpt` to dissect the forgetting on ResNet.  
* Step 1: Move your encoder model files (.ckpt) to `CFD/Pytorch_Models/` to visualize your models. 
* Step 2: Change the argument "basenet" to specify the reference model (the model does not show forgetting - M19) and "img_shape" to your input image shape of your model in "run_CFD.py".

For example, we define basenet is M19.ckpt, then we can also add M20.ckpt and M24.ckpt to `CFD/Pytorch_Models/` to run CFD.

## Tested data
In our experiments, tested images are downloaded from MS-COCO dataset.
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
