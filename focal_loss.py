# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by chen shuai
# --------------------------------------------------------

"""
Focal loss 
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, gamma):
        super(FocalLossOperator, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma

    def forward(self, is_train, req, in_data, out_data, aux):

        cls_score = in_data[0]
        self._cls_score = cls_score
        labels = in_data[1].asnumpy()
        self._labels = labels

        pro_ = mx.nd.SoftmaxActivation(cls_score) + 1e-14
        pro_ = pro_.asnumpy()
        # restore pt for backward
        self._pt = pro_[np.arange(pro_.shape[0],dtype = 'int'), labels.astype('int')]

        ### note!!!!!!!!!!!!!!!!
        # focal loss value is not used in this place we should forward the cls_pro in this layer, the focal vale should be calculated in metric.py
        # the method is in readme
        #  focal loss (batch_size,num_class)
        loss_ = -1 * np.power(1 - pro_, self._gamma) * np.log(pro_)

        self.assign(out_data[0],req[0],mx.nd.array(pro_))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        cls_score = self._cls_score
        labels = self._labels
        pro_ = mx.nd.SoftmaxActivation(cls_score) + 1e-14
        pro_ = pro_.asnumpy()
        #i!=j
        pt = self._pt
        dx = np.power(1 - pro_, self._gamma - 1)
        pt = pt.reshape(len(pt),1)
        dx = np.power(1 - pt, self._gamma - 1) *(self._gamma * (-1 * pt * pro_) * np.log(pro_) + pt * (1 - pro_))
        ####i==j 
        #reload pt
        pt = self._pt 
        dx[np.arange(pro_.shape[0],dtype = 'int'), labels.astype('int')] = np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt -1) 
        self.assign(in_grad[0], req[0], mx.nd.array(dx))

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, gamma):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._gamma = float(gamma)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return  [data_shape, labels_shape],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._num_classes, self._gamma)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
