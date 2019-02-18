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

# pylint: disable=arguments-differ, too-many-lines
# coding: utf-8
"""Definition of various recurrent neural network cells."""
__all__ = ['EventHandler','LoggingHandler','CheckpointHandler','MetricHandler']
import logging
import os
import time

class EventHandler(object):
    def __init__(self, estimator):

        self._estimator= estimator

    def train_begin(self):
        pass
    def train_end(self):
        pass

    def batch_begin(self):
        pass

    def batch_end(self):
        pass

    def epoch_begin(self):
        pass

    def epoch_end(self):
        pass


class LoggingHandler(EventHandler):
    """Basic Logging Handler that applies to every Gluon estimator by default.

    """
    def __init__(self, estimator, log_name=None, file_name=None, file_location=None,):
        super(LoggingHandler, self).__init__(estimator)
        log_name = log_name or 'Gluon Estimator'
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        streamhandler = logging.StreamHandler()
        self.logger.addHandler(streamhandler)
        # save logger to file only if file name or location is specified
        if file_name or file_location:
            file_name = file_name or log_name or 'estimator_log'
            file_location = file_location or './'
            filehandler = logging.FileHandler(os.path.join(file_location, file_name))
            self.logger.addHandler(filehandler)


    def train_begin(self):
        pass
        #logger.info(opt)
    def train_end(self):
        pass

    def batch_begin(self):
        self.batch_start = time.time()

    def batch_end(self):
        batch_time = time.time() -  self.batch_start
        epoch = self._estimator.train_stats['epochs'][-1]
        step = self._estimator.train_stats['step']
        msg = '[Epoch %d] [Step %s] time/step: %.3fs ' % (epoch, step, batch_time)
        for key in self._estimator.train_stats.keys():
            if key.startswith('batch_'):
                msg += key[6:] + ': ' + '%.4f ' %self._estimator.train_stats[key]
        self.logger.info(msg)

    def epoch_begin(self):
        self.epoch_start = time.time()

    def epoch_end(self):
        epoch_time = time.time() -  self.epoch_start
        epoch = self._estimator.train_stats['epochs'][-1]
        msg = 'Epoch %d finished in %.3fs: ' % (epoch, epoch_time)
        for key in self._estimator.train_stats.keys():
            if key.startswith('train_') or key.startswith('test_'):
                msg += key + ': ' + '%.4f ' %self._estimator.train_stats[key][epoch]
        self.logger.info(msg)


class CheckpointHandler(EventHandler):
    def __init__(self,estimator,  whenToCheckpoint =5, ckpt_loc='./', filename = 'my_model',hybridise=False):
        super(CheckpointHandler,self).__init__(estimator)
        #self._estimator= estimator
        # estimator._train_stats = {"lr" : 0.1, "train_acc" : [0.85], "val_acc" :[0.99]}
        self._hybridise = hybridise
        self.ckpt_loc= ckpt_loc
        #self._best_score=
        self._whenToCheckpoint=  whenToCheckpoint
        self._filename = filename
    def train_begin(self):
        if self._hybridise:
            self._estimator._net.hybridise()
    def train_end(self):
        if self._hybridise:
            train_metric_name, train_metric_val = zip(*(self._estimator._metric.get_name_value()))

            for names in train_metric_name:
                train_metric_score = self._estimator._train_stats['train_'+names][-1]
                self._estimator._net.export('%s/%.4f-best' % (self.ckpt_loc, train_metric_score), self._estimator._epoch)
        else:
            self._estimator._net.save_parameters('%s/imagenet-%d.params' % (self.ckpt_loc,  self._estimator._epoch))
            #self._estimator._trainer.save_states('%s/imagenet-%d.states' % (self.ckpt_loc,  self._estimator._epoch))


    def batch_begin(self):
        pass

    def batch_end(self):
        pass

    def epoch_begin(self):
        pass

    def epoch_end(self):
        print("called checkpointing")
        if (self._estimator._epoch+1)%self._whenToCheckpoint ==0 :
            train_metric_name, train_metric_val = zip(*(self._estimator._metric.get_name_value()))
            for names in train_metric_name:
                train_metric_score = self._estimator._train_stats['train'+names][-1]
                self._estimator._net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params' % (self.ckpt_loc, train_metric_score, self._filename, self._estimator._epoch))
                #self._estimator._trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states' % (self.ckpt_loc, train_metric_score, self._filename, self._estimator._epoch))

        ##move to earlystopping
        #if err_top1_val < best_val_score:
        #    best_val_score = err_top1_val
        #    self._estimator._net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params' % (self.ckpt_loc, best_val_score, self._filename, self._estimator._epoch))
        #    self._estimator._trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states' % (self.ckpt_loc, best_val_score, self._filename, self._estimator._epoch))

class MetricHandler(EventHandler):
    def __init__(self,estimator):
        super(MetricHandler,self).__init__(estimator)
        #self._estimator= estimator
        # estimator._train_stats = {"lr" : 0.1, "train_acc" : [0.85], "val_acc" :[0.99]}
        self._metric= None
        self._valdataloader = None

    def evaluate_loss_and_metric(self, dataloader):
        for (i,batch) in enumerate(dataloader):
            X= batch[0]
            y= batch[1]
            y_pred = self._estimator._net(X)
            ##replace this with custom function just as discussed
            ## Also update val_loss things
            for metrics in self._metric:
                metrics.update(y_pred,y)

        for metrics in self._metric:
            metric_eval= metrics.get_name_value()
            for name, val in metric_eval:
                self._estimator._train_stats['val_'+ name].append(val)
        ##what should be return type
        return

    def train_begin(self):
        self._metric= self._estimator._metric
        for metrics in self._metric:
            train_metric_name, train_metric_val = zip(*(metrics.get_name_value()))
            for m_names in train_metric_name:
                # print(self._metric.get()[0])
                # print(m_names)
                self._estimator._train_stats['train_'+m_names] = []
                self._estimator._train_stats['val_' + m_names] = []
    def train_end(self):
        pass

    def batch_begin(self):
        pass

    def batch_end(self):
        ##if mapping doesnt exist raise error size(metrics) not equal to size(labels)
        for metrics in self._metric:
            ##TODO: deal it with a separate update functionsn to take care of mapping metric to outputs- use same for eval
            metrics.update(self._estimator.y, self._estimator.y_hat)

    def epoch_begin(self):
        for metrics in self._metric:
            metrics.reset()

    def epoch_end(self):
        print("metrci")
        for metrics in self._metric:
            metric_val = metrics.get_name_value()
            for name, val in metric_val:
                self._estimator._train_stats['train_'+name].append(val)

        ##get validation metrics
        if self._estimator._epoch % self._estimator._evaluate_every == 0:
            self.evaluate_loss_and_metric(self._valdataloader)

        ##move to earlystopping
        #if err_top1_val < best_val_score:
        #    best_val_score = err_top1_val
        #    self._estimator._net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params' % (self.ckpt_loc, best_val_score, self._filename, self._estimator._epoch))
        #    self._estimator._trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states' % (self.ckpt_loc, best_val_score, self._filename, self._estimator._epoch))
