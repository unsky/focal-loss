# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by chenshuai
# --------------------------------------------------------

"""
focal loss with softmax
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

        cls_score    = in_data[0]
        bbox_pred    = in_data[1]
        labels       = in_data[2].asnumpy()
        bbox_targets = in_data[3]
        bbox_weights = in_data[4]

        per_roi_loss_cls = mx.nd.SoftmaxActivation(cls_score) + 1e-14
        per_roi_loss_cls = per_roi_loss_cls.asnumpy()
        per_roi_loss_cls = per_roi_loss_cls[np.arange(per_roi_loss_cls.shape[0], dtype='int'), labels.astype('int')]
        per_roi_loss_cls = -1 * np.log(per_roi_loss_cls)
        per_roi_loss_cls = np.reshape(per_roi_loss_cls, newshape=(-1,))

        per_roi_loss_bbox = bbox_weights * mx.nd.smooth_l1((bbox_pred - bbox_targets), scalar=1.0)
        per_roi_loss_bbox = mx.nd.sum(per_roi_loss_bbox, axis=1).asnumpy()

        top_k_per_roi_loss = np.argsort(per_roi_loss_cls + per_roi_loss_bbox)
        labels_ohem = labels
        labels_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = -1
        bbox_weights_ohem = bbox_weights.asnumpy()
        bbox_weights_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = 0

        labels_ohem = mx.nd.array(labels_ohem)
        bbox_weights_ohem = mx.nd.array(bbox_weights_ohem)

        for ind, val in enumerate([labels_ohem, bbox_weights_ohem]):
            self.assign(out_data[ind], req[ind], val)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, gamma):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._gamma = float(gamma)



    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        return in_shape, \
               [data_shape, labels_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._num_classes, self.gama)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
