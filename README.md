[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## The implemntation of FCN on Geological data

- The model is based on CVPR '15 best paper honorable mentioned [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

## Results
### RGB bands
#### Ground truth
<p float="left">
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_24_10/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_31_07/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_35_15/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_36_30/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_37_37/target.png' padding='5px' height="150px"></img>
</p>


#### Prediction
<p float="left">
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_24_10/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_31_07/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_35_15/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_36_30/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results/32N-26E-224N_37_37/prediction.png' padding='5px' height="150px"></img>
</p>

### Mulrispectral bands
#### Ground truth
<p float="left">
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_16_34/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_18_12/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_21_07/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_26_14/target.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_27_35/target.png' padding='5px' height="150px"></img>
</p>


#### Prediction
<p float="left">
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_16_34/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_18_12/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_21_07/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_26_14/prediction.png' padding='5px' height="150px"></img>
  <img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='Test_data/results_multispectral/32N-26E-224N_27_35/prediction.png' padding='5px' height="150px"></img>
</p>


## Performance

I train with custom satellite images dataset

|    Bands    | Pixel Accuracy|Best IOU|
|-------------|:-------------:|-------:|
|Multispectral|      90%      |  0.57  |
|     RGB     |      70%      |  0.52  |

## Training

### Install packages
```bash
pip install -r requirements.txt
```

### Run the code
- Train with RGB band

```python
python train_fcn.py --data-dir "<Path to data directory>" --num-epochs 50 --use-pretrained False --checkpoint-dir "<Path to checkpoint directory if pretrained is true>"
```

- Train with RGB band

```python
python train_fcn_multispectral.py --data-dir "<Path to data directory>" --num-epochs 50 --use-pretrained False --checkpoint-dir "<Path to checkpoint directory if pretrained is true>"
```

## Inference
- Infer with default 10 samples

```python
python inference_rgb --data-dir "<Path to test data directory>" --results-dir "<Path to result directory>" --test-count 10 --checkpoint-dir "<Path to checkpoint directory if pretrained is true>"
```

```python
python inference_multispectral --data-dir "<Path to test data directory>" --results-dir "<Path to result directory>" --test-count 10 --checkpoint-dir "<Path to checkpoint directory if pretrained is true>"
```