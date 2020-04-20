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

Move your encoder models to "Pytorch_Models" (.ckpt) to visualize your models. Change the arguement "basenet" to the reference model (the model does not show forgetting) and "img_shape" to your model input shape in "run_DeepVis.py".

## Data
Put the tested images (.jpg) and ground truth numpy files (.npy) to the folder ./data

## Groud truth numpy files
Download the Train/Val annotations from MS-COCO download page (http://cocodataset.org/#download) and use their API (https://github.com/cocodataset/cocoapi) to generate ground truth numpy files.

## Run
Simply run
```
python run_CFD.py
```

## Results
The IoU results and forgetting report will be generated at "./IoU_results". The result format is [test_image]_[test_model]_[reference]_[visualized block]
e.g. airplane_M19_GT_block1 shows the IoU between vision of the first block of model "M19" and "Ground Truth" tested on image "airplane".
