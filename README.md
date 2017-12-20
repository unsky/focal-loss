# focal-loss

The code is unofficial version for `focal loss for Dense Object Detection`.
 https://arxiv.org/abs/1708.02002

this is implementtd using mxnet python layer.

The retina-net is in https://github.com/unsky/RetinaNet



# usage
Assue that you have put the focal_loss.py in your operator path

you can use:

```
from your_operators.focal_loss import *

cls_prob = mx.sym.Custom(op_type='FocalLoss', name = 'cls_prob', data = cls_score, labels = label, alpha =0.25, gamma= 2)

```

# focal loss with softmax on kitti(10 cls)
this is my experiments on kitti 10 cls, the performance on hard cls is great!!

| method@0.7                           | car           | van   | Truck |cyclist |pedestrian|person_sitting|tram  |misc  |dontcare|
| -------------                        |:-------------:| -----:| -----:| ------:|---------:|-------------:|-----:|-----:|-------:|
| base line(faster rcnn + ohem(1:2))   |      0.7892   |0.7462 |0.8465 |0.623   |0.4254    |0.1374        |0.5035|0.5007|0.1329  |
| faster rcnn + focal loss with softmax|      0.797    |0.874  | 0.8959|0.7914  |0.5700    |0.2806        |0.7884|0.7052|0.1433  |

![image](https://github.com/unsky/focal-loss/blob/master/readme/res.png)

#### about parameters in this expriment
https://github.com/unsky/focal-loss/issues/5


# note!!

## very important!!!

~~in my experiment, i have to use the strategy in  `paper section 3.3`.~~

~~LIKE:~~

![image](https://github.com/unsky/focal-loss/blob/master/readme/loss1.png)

~~Uder such an initialization, in the presence of class imbalance, the loss due to the frequent class can dominate total loss and cause instability in early training.~~
 



~~##AND YOU CAN TRY MY INSTEAD STRATEGY:~~

~~train the model using the classical softmax for several times (for examples 3 in kitti dataset)~~

~~choose a litti learning rate:~~

~~and the traing loss will work well:~~

![image](https://github.com/unsky/focal-loss/blob/master/readme/loss2.png)
## about alpha

https://github.com/unsky/focal-loss/issues/4

## now focal loss with softmax work well


focal loss value is not used in focal_loss.py, becayse we should forward the cls_pro in this layer,
the major task of focal_loss.py is to backward the focal loss gradient.

the focal loss vale should be calculated in metric.py and  use normalization in it.

and this layer is not support `use_ignore`

for example :

```python
class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        gamma = 2
        alpha = 0.25

        cls_loss = alpha*(-1.0 * np.power(1 - cls, gamma) * np.log(cls))

        cls_loss = np.sum(cls_loss)/len(label)
        #print cls_loss
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

```
# the value must like
## forward value
![image](https://github.com/unsky/focal-loss/blob/master/readme/forward.png)
## backward gradient value
![image](https://github.com/unsky/focal-loss/blob/master/readme/back_cure.png)

you can check the gradient value in your debug(if need).
By the way

this is my derivation about backward, if it has mistake, please note to me.

# softmax activation:

![image](https://github.com/unsky/focal-loss/blob/master/readme/2.jpg)

# cross entropy with softmax

![image](https://github.com/unsky/focal-loss/blob/master/readme/3.jpg)

# Focal loss with softmax

![image](https://github.com/unsky/focal-loss/blob/master/readme/1.jpg)


