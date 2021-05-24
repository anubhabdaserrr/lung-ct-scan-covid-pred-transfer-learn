# Diagnosing Covid-19 from Lung CT Scans using transfer learning (VGG16, DenseNet121, InceptionV3, etc.)

**Project Report by Anubhab Das** 

Date : 27th April, 2021

[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/anubhabdaserrr/lung-ct-scan-covid-pred-transfer-learn/blob/main/lung_ct_scan_covid_pred_nb.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anubhabdaserrr/lung-ct-scan-covid-pred-transfer-learn/blob/main/lung_ct_scan_covid_pred_nb.ipynb)

## Objective & data

In this project, I fine-tuned a bunch of pre-trained models like VGG16, InceptionV3, etc. for diagnosing Covid-19 based on Lung CT Scans. The dataset used can be found [here](https://www.kaggle.com/luisblanche/covidct). The inspiration behind this project was primarily to speed-up the diagnostic process by building detection models trained on CT scans which are non-invasive in nature.

**Note : This project is built purely for educational purposes and not for medical diagnosis.**

![](./misc/imgs_ct_scans.png)

Transfer learning is basically using the knowledge acquired while finding the solution to a certain problem to solve other problems. Putting it in context of computer vision models using deep convolutional neural networks, when a model is trained on a large dataset, the initial (convolutional) layers learn to detect general features pertaining to image understanding - Like detecting edges followed by composition of these edges and so on. The initial layers serve as feature detectors which are built on top of each other in a hierarchical fashion to obtain more complex features as we go deeper into the network. Because these networks learns to detect general patterns in images, we can "transfer" this learned knowledge to other datasets to faciliate learning. This is particularly useful when our target dataset is small as we can use prior knowledge i.e. pretrained neural networks and tune it to the target dataset. This is done by retraining the final (or the last few) fully connected layers of the network since they're responsible for classification or by adding a few fully connected layers (if need be).

![](./misc/transfer_learn_net.png)

### Train & Test Distribution

![](./misc/train_distrib.png) ![](./misc/test_distrib.png)

As is evident from the diagrams, both train & test distributions are almost uniform/ balanced.

## Model & Training details

### Models used 
A vanilla CNN (BaselineNet) has been used to serve the purpose of a Baseline Model against which all the other models will be evaluated.

| Models         	| Description 	| 
|----------------	|----------	|
| BaselineNet    	| Vanilla ConvNet based Baseline model     	|
| VGG16          	| Upscaled version of AlexNet (which the first CNN based winner of ImageNet.) in terms of layers.	|
| DenseNet121    	| Made up of dense blocks where each layer is connected to every other layer in a feedforward way. Advantages :  alleviates vanishing-gradient problem, strengthens feature propagation, encourages feature reuse, reduction in no. of parameters. 50-layers Densenet outperforms 152-layers ResNet.  	|
| ResNet50       	| Makes use of skip connections to deal with vanishing gradients as well as to allow the later layers to learn from the information learned by the initial layers     	|
| InceptionV3    	| Makes use of Inception modules as building blocks to reduce compute, which contain multiple branches of Conv. & Pool layers which are concatenated channel-wise prior to feeding it to the subsequent blocks. Also uses auxiliary classifiers as regularizers.    	|
| EfficientNetB3 	| 0.70     	|

## Evaluation & test performance

### Model Comparsion | Metric Values : 

| Models         	                  | Accuracy 	| Recall / Sensitivity 	| F1 Score 	| AUC   	|
|----------------	                  |----------	|----------------------	|----------	|-------	|
| BaselineNet    	                  | 0.75     	| **0.94**                 	| 0.790    	| 0.894 	|
| VGG16          	                  | 0.86     	| 0.84                 	| 0.857    	| 0.909 	|
| DenseNet121    	                  | **0.88**     	| 0.82                 	| **0.872**    	| **0.932** 	|
| ResNet50       	                  | 0.87     	| 0.82                 	| 0.863    	| 0.898 	|
| InceptionV3 with retrained BN    	| 0.82     	| 0.82                 	| 0.820    	| 0.881 	|
| EfficientNetB3 with retrained BN 	| 0.70     	| 0.56                 	| 0.651    	| 0.805 	|

DenseNet121 outperforms all models for accuracy, F1 score & AUC metrics. The Baseline model has the highest recall value. Overall, pre-trained models seem to work pretty well in most cases with regards to the aforementioned metrics except for recall. Recall or sensitivity is a very important metric when dealing with diagnostic tests because false negatives are much more unfavourable compared to false positives - The consequence of misdiagnosing a patient who actually has the condition could be catastrophic. However, false positives are not as harmful. However we're using a balanced dataset, therefore accuracy is a good enough metric. The Baseline model has a high recall and low accuracy. This is indicative of the fact that our baseline model might be biased towards classifying test cases as positive even if they're not. This doesn't particularly seem useful i.e. our baseline model might be closer to a naive model compared to most other models. In this case, while choosing the best performing model it might make sense to jointly consider accuracy and recall values.

EfficientNetB3 seems to be the worst performing model out of all of the models.

The test set is pretty small - 50 instances in each class. Therefore, these numbers might not be a good estimate for out-of-sample model performance.

**Note : Performance of InceptionV3 & EfficientNetB3 were very close to no skill classifiers (i.e. 0.5 out-of-sample accuracy) if the BatchNorm layers were kept frozen. Retraining them helped with improving model performance.**

### Plot of metrics vs no. of parameters in models :

The purpose of this section is to study whether pretrained models actually help us re-use learned feature detectors to solve a different problem. This, in and of itself, seems like a daunting task hence we make a very preliminary attempt at tackling this problem.

<img src="./misc/metrics_trainable_params_plot.png"/> 
<img src="./misc/metrics_total_params_plot.png"/>


## Limitations & future directions 
limitations

## References
Links go here
