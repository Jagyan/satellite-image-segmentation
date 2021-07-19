[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## The implemntation of FCN on Geological data

- The model is based on CVPR '15 best paper honorable mentioned [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

## Results
### Trials
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/trials.png' padding='5px' height="150px"></img>

### Training Procedures
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/result.gif' padding='5px' height="150px"></img>


## Performance

I train with custom satellite images dataset

|dataset|n_class|pixel accuracy|bands|
|---|---|---|---
|Satellite images|3|91%|Multispectral
|Satellite images|3|81%|RGB

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