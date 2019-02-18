# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""Gluon Estimator"""


import warnings

from .event_handler import LoggingHandler
from ... import *
from ... import gluon, autograd
from ...metric import EvalMetric, Loss

__all__ = ['Estimator']


class Estimator(object):
    """
    Estimator Class for easy model training
    TODO: update doc
    """

    def __init__(self, net,
                 loss=None,
                 metrics=None,
                 initializer=None,
                 trainers=None,
                 ctx=None):

        self.net = net
        if isinstance(loss, gluon.loss.Loss):
            self.loss = [loss]
        else:
            self.loss = loss or []
        if isinstance(metrics, EvalMetric):
            self.metrics = [metrics]
        else:
            self.metrics = metrics or []

        self.initializer = initializer
        # store training statistics
        self.train_stats = {}
        self.train_stats['epochs'] = []
        self.train_stats['learning_rate'] = []
        # time used for each epoch
        self.train_stats['step'] = ''
        for metric in self.metrics:
            # record a history of metrics over each epoch
            self.train_stats['train_' + metric.name] = []
            # only record the latest metric numbers after each batch
            self.train_stats['batch_' + metric.name] = 0.
        self.loss_metrics = []
        # using the metric wrapper for loss to record loss value
        for loss in self.loss:
            self.loss_metrics.append(Loss(loss.name))
            self.train_stats['train_' + loss.name] = []
            # only record the latest loss numbers after each batch
            self.train_stats['batch_' + loss.name] = 0.

        # TODO：discuss whether to 1) hide trainer logic from user 2) user has to initialize net and trainer before estimator
        # initialize the net if no initializer specified
        if not self.initializer:
            # no force reinitialize in case net is already initialized outside fit method
            self.net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=False)
        else:
            # initialize with user specified initializer
            self.net.initialize(init=self.initializer, ctx=ctx, force_reinit=True)

        if isinstance(ctx, Context):
            self.ctx = [ctx]
        if not ctx:
            if context.num_gpus() > 0:
                # only use 1 GPU by default
                self.ctx = [context.gpu(0)]
            else:
                self.ctx = [context.cpu()]
        if isinstance(trainers, gluon.Trainer):
            self.trainers = [trainers]
        else:
            self.trainers = trainers or []
        if not self.trainers:
            warnings.warn("No trainer specified, default SGD optimizer with learning rate 0.001 is used.")
            self.trainers = [gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.001})]

    def _batch_fn(self, batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    def fit(self, train_data,
            val_data=None,
            epochs=1,
            batch_size=None,
            event_handlers=None):

        if not batch_size:
            batch_size = 32 * len(self.ctx)

        event_handlers = event_handlers or []
        if not event_handlers:
            event_handlers.append(LoggingHandler(self))

        # TODO: handle validation logic and update train stats
        do_validation = False
        if val_data:
            do_validation = True

        # training begin
        for handler in event_handlers:
            handler.train_begin()

        for epoch in range(epochs):
            # epoch begin
            self.train_stats["epochs"].append(epoch)
            self.train_stats["learning_rate"].append(self.trainers[0].learning_rate)

            for handler in event_handlers:
                handler.epoch_begin()

            for metric in self.metrics + self.loss_metrics:
                metric.reset()

            for i, batch in enumerate(train_data):
                data, label = self._batch_fn(batch, self.ctx)

                # batch begin
                for handler in event_handlers:
                    handler.batch_begin()

                with autograd.record():
                    pred = [self.net(x) for x in data]
                    losses = []
                    for loss in self.loss:
                        losses.append([loss(y_hat, y) for y_hat, y in zip(pred, label)])

                for loss in losses:
                    for l in loss:
                        l.backward()

                # update metrics
                for metric in self.metrics:
                    metric.update(label, pred)
                    self.train_stats['batch_' + metric.name] = metric.get()[1]
                for loss, loss_metric, in zip(losses, self.loss_metrics):
                    loss_metric.update(0, [l for l in loss])
                    self.train_stats['batch_' + loss_metric.name] = loss_metric.get()[1]

                self.train_stats['step'] = str(batch_size * (i + 1)) + '/' + str(len(train_data._dataset))

                for trainer in self.trainers:
                    trainer.step(batch_size)

                # batch end
                for handler in event_handlers:
                    handler.batch_end()

            for metric in self.metrics + self.loss_metrics:
                self.train_stats['train_' + metric.name].append(metric.get()[1])
            # epoch end
            for handler in event_handlers:
                handler.epoch_end()

        # train end
        for handler in event_handlers:
            handler.train_end()
