# focal-loss

The code is unofficial version for `focal loss for Dense Object Detection`.
 https://arxiv.org/abs/1708.02002

this is implementtd using mxnet python layer.


# usage
Assue that you have put the focal_loss.py in your operator path

you can use:

```
from your_operators.focal_loss import *

cls_prob = mx.sym.Custom(op_type='FocalLoss',name='cls_prob', data=cls_score, labels=label, gamma= 2)

```


# note!!

## very import!!!

in my experiment, i have to use the strategy in  `paper section 3.3`.

Uder such an initialization, in the presence of class imbalance, the loss due to the frequent class can dominate total loss and cause instability in early training.
 
focal loss value is not used in focal_loss.py, becayse we should forward the cls_pro in this layer,
the major task of focal_loss.py is to backward the focal loss gradient.


the focal vale should be calculated in metric.py and  use normalization in it.


and this layer is not support `use_ignore`

for example :

```
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
        cls_loss = (-1.0 * np.power(1 - cls, 2) * np.log(cls))

        cls_loss= cls_loss/np.sum(cls_loss)
        #cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]
```
By the way

this is my derivation about backward, if it has mistake, please note to me.

# softmax activation:

![image](https://github.com/unsky/focal-loss/blob/master/2.jpg)

# cross entropy with softmax

![image](https://github.com/unsky/focal-loss/blob/master/3.jpg)

# Focal loss with softmax

![image](https://github.com/unsky/focal-loss/blob/master/1.jpg)


