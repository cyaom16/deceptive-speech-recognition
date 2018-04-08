from sklearn.metrics import roc_auc_score
import tensorflow as tf
from util import *


class Config:
    """Configuration parameters"""
    rnn_type = 'gru'
    direction = 'uni'
    num_feats = 13
    num_units = 32
    num_layers = 2
    num_classes = 2

    num_epochs = 50
    batch_size = 128
    keep_prob = 0.5
    lr = 1e-3
    # L2-regularization factor
    beta = 0.01


class Polygraph(object):
    """
    Main model that constructs TF graph and perform training and validation

    Args:
        direction:  Choice of unidiretional or bidirection RNN
        num_units:  Number of hidden units on a RNN layer
        num_layers: Number of RNN layers
        lr:         Learning rate

    Return:
        Model object
    """

    def __init__(self, num_units, num_layers, rnn_type='lstm', direction='uni', lr=1e-3):
        self.inputs = None
        self.targets = None
        self.seq_lens = None
        self.keep_prob = None

        self._num_units = num_units
        self._num_layers = num_layers
        self._rnn_type = rnn_type
        self._direction = direction

        self.logits = None
        self.loss = None
        self.predictions = None
        self.lr = lr
        self.optimizer = None
        self.accuracy = None

        # self._build_graph()

    def _add_placeholders(self):
        """Add placeholders"""
        self.inputs = tf.placeholder(tf.float32, [None, None, Config.num_feats], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, Config.num_classes], name='targets')
        self.seq_lens = tf.placeholder(tf.int32, [None], name='seq_lens')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.pos_weight = tf.placeholder(tf.float32, name='pos_weight')  # Positive-class weight

    def _add_feed_dict(self, X_batch, y_batch, l_batch, keep_prob, pos_weight):
        """Add feed_dict"""
        feed_dict = {self.inputs: X_batch,
                     self.targets: y_batch,
                     self.seq_lens: l_batch,
                     self.keep_prob: keep_prob,
                     self.pos_weight: pos_weight}
        return feed_dict

    def _add_prediction_op(self):
        """Construct network and prediction op"""

        # RNN cell definition helper
        def rnn_cell(num_units, keep_prob, rnn_type='lstm'):
            if rnn_type == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(num_units)
            elif rnn_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(num_units)
            elif rnn_type == 'rnn':
                cell = tf.nn.rnn_cell.RNNCell(num_units)
            else:
                raise ValueError("Unknown specified rnn_type '{}'".format(rnn_type))
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # For sequence classification, we need the very last relevant output that supposes to
        # contain the entire information of the sequence from the history
        if self._direction == 'uni':
            # Unidirectional (stacked if layers > 1)
            cell = [rnn_cell(self._num_units, 1., rnn_type=self._rnn_type)
                    for _ in range(self._num_layers)]
            network = tf.nn.rnn_cell.MultiRNNCell(cell)

            # 'outputs' is of shape [batch_size, timesteps, num_units]
            outputs, _ = tf.nn.dynamic_rnn(network,
                                           self.inputs,
                                           sequence_length=self.seq_lens,
                                           dtype=tf.float32)

            # Extract the last relevant output from dynamic run using variable-length sequences
            last = self._extract_relevant(outputs, self.seq_lens - 1)

        elif self._direction == 'bi':
            # Bidirectional
            cell_fw = rnn_cell(self._num_units, self.keep_prob, rnn_type=self._rnn_type)
            cell_bw = rnn_cell(self._num_units, self.keep_prob, rnn_type=self._rnn_type)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                         cell_bw,
                                                         self.inputs,
                                                         sequence_length=self.seq_lens,
                                                         dtype=tf.float32)
            # Unpack forward/backward outputs
            output_fw, output_bw = outputs

            # Not sure why cannot use the staticmethod here!
            # last_fw = self._extract_relevant(output_fw, self.seq_lens - 1)
            batch_range = tf.range(tf.shape(output_fw)[0])
            indices = tf.stack([batch_range, self.seq_lens - 1], axis=1)
            last_fw = tf.gather_nd(output_fw, indices)

            # The last output from the backward layer locates at the first timestep
            last_bw = output_bw[:, 0, :]
            last = tf.concat([last_fw, last_bw], 1)

        else:
            raise ValueError("Unknown specified direction '{}'".format(self._direction))

        # Apply ReLu
        relu = tf.nn.relu(last, name='relu')

        # Apply dropout
        drop = tf.nn.dropout(relu, self.keep_prob)

        # For binary classification, we just need one unit from the fully-connected
        # Weights are initialized using truncated normal initializer
        # Biases are initialized with non-zero for layer pre-activation
        self.logits = tf.layers.dense(drop,
                                      units=Config.num_classes,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name='dense')
        # self.predictions = tf.cast(tf.greater(self.logits, 0.), tf.float32)
        self.predictions = tf.nn.softmax(self.logits)

    def _add_loss_op(self):
        """Add loss"""
        # For binary classification, we need sigmoid cross-entropy loss
        # ce = tf.nn.sigmoid_cross_entropy_with_logits(self.targets, self.logits)
        ce = tf.nn.weighted_cross_entropy_with_logits(targets=self.targets,
                                                      logits=self.logits,
                                                      pos_weight=self.pos_weight)
        # For TF 1.4+, use tf.nn.softmax_cross_entropy_with_logits_v2
        # ce = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets,
        #                                              logits=self.logits)
        self.loss = tf.reduce_mean(ce)

        # L2-regularization
        # with tf.variable_scope('dense', reuse=True):
        #     weights = tf.get_variable('kernel')
        # l2_penalty = tf.nn.l2_loss(weights, name='l2-penalty')
        # self.loss = tf.reduce_mean(self.loss + Config.beta * l2_penalty)

    def _add_metric_op(self):
        """Add evaluation metric"""
        # correct = tf.equal(self.targets, self.predictions)
        correct = tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.predictions, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def _add_training_op(self):
        """Add training op"""
        train_op = tf.train.AdamOptimizer(self.lr)
        # self.optimizer = train_op.minimize(self.loss)

        # Gradient clipping
        gradients = train_op.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.optimizer = train_op.apply_gradients(capped_gradients)

    def _add_summary_op(self):
        """Merge summary"""
        self.merge_summary = tf.summary.merge_all()

    def build_graph(self):
        """Construct the computational graph"""
        self._add_placeholders()
        self._add_prediction_op()
        self._add_loss_op()
        self._add_metric_op()
        self._add_training_op()
        # self._add_summary_op()

    def run(self, session, X_batch, y_batch, l_batch, keep_prob, pos_weight=1, train=True):
        """
        Call to run

        Args
            session: Session object
            X_batch: A batch of inputs
            y_batch: A batch of targets
            l_batch: A batch of sequence lengths
            keep_prob: Dropout (keep) probabilities
            pos_weight: Positive-class weight
            train: Training flag

        Returns
            loss, accuracy, predictions
        """
        feed = self._add_feed_dict(X_batch, y_batch, l_batch, keep_prob, pos_weight)

        if train:
            _ = session.run([self.optimizer], feed)

        loss, accuracy, pred = session.run([self.loss, self.accuracy, self.predictions], feed)

        return loss, accuracy, pred

    @staticmethod
    def _extract_relevant(outputs, indices):
        """
        Extract the relevant output when using dynamic_rnn with variable-length sequence inputs

        Args:
            outputs: Tensor that will be subsetted.
            indices: Index to take (one for each element along axis 0 of data).
        Returns:
            A subsetted tensor.
        """
        batch_range = tf.range(tf.shape(outputs)[0])
        indices_nd = tf.stack([batch_range, indices], axis=1)

        return tf.gather_nd(outputs, indices_nd)


class PolygraphCRNN(Polygraph):
    """
    Main model that constructs TF graph and perform training and validation

    Args:
        num_units:  Number of hidden units on a RNN layer
        num_layers: Number of RNN layers
        rnn_type:   Type of RNN from 'lstm', 'gru', 'rnn', i.e., LSTM, GRU, regular RNN
        direction:  Choice of uni-diretional or bi-direction RNN
        lr:         Learning rate

    Return:
        Model object
    """

    def __init__(self, num_units, num_layers, rnn_type='lstm', direction='uni', lr=1e-3):
        super(PolygraphCRNN, self).__init__(num_units,
                                            num_layers,
                                            rnn_type=rnn_type,
                                            direction=direction,
                                            lr=lr)

    def _add_prediction_op(self):
        """
        Construct network and prediction op
        this method overwrites the parent class
        """
        # CNN 3 layers of Conv2D + MaxPooling
        max_time = tf.shape(self.inputs)[1]

        inputs_cnn = tf.reshape(self.inputs, [-1, max_time, Config.num_feats, 1])

        # 1st layer of Conv2D + MaxPooling
        conv_1 = tf.layers.conv2d(inputs=inputs_cnn,
                                  filters=16,
                                  kernel_size=[2, 2],
                                  padding='same',
                                  activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(inputs=conv_1,
                                         pool_size=[2, 2],
                                         strides=2)
        # 2nd layer of Conv2D + MaxPooling
        conv_2 = tf.layers.conv2d(inputs=pool_1,
                                  filters=32,
                                  kernel_size=[2, 2],
                                  padding='same',
                                  activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(inputs=conv_2,
                                         pool_size=[2, 2],
                                         strides=2)
        # 3rd layer of Conv2D + MaxPooling
        conv_3 = tf.layers.conv2d(inputs=pool_2,
                                  filters=64,
                                  kernel_size=[2, 2],
                                  padding='same',
                                  activation=tf.nn.relu)
        pool_3 = tf.layers.max_pooling2d(inputs=conv_3,
                                         pool_size=[2, 2],
                                         strides=2)

        batch_size = tf.shape(pool_3)[0]
        # Timesteps has been reduced to 1/8 of original max_time due to 3 levels of pooling
        reduced_steps = tf.shape(pool_3)[1]
        # 'inputs_rnn' is of shape (batch_size, reduced_teps, num_filters=64)
        inputs_rnn = tf.reshape(pool_3, [batch_size, reduced_steps, 64])
        # Sequence length reduced to 1/8 as well
        reduced_lens = tf.cast(tf.divide(self.seq_lens, 8), tf.int32)

        # RNN cell definition helper
        def rnn_cell(num_units, keep_prob, rnn_type='lstm'):
            if rnn_type == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(num_units)
            elif rnn_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(num_units)
            elif rnn_type == 'rnn':
                cell = tf.nn.rnn_cell.RNNCell(num_units)
            else:
                raise ValueError("Unknown specified rnn_type '{}'".format(rnn_type))

            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # For sequence classification, we need the very last relevant output that supposes to
        # contain the entire information of the sequence from the history
        if self._direction == 'uni':
            # Unidirectional (stacked if layers > 1)
            cell = [rnn_cell(self._num_units, 1., rnn_type=self._rnn_type)
                    for _ in range(self._num_layers)]
            network = tf.nn.rnn_cell.MultiRNNCell(cell)

            # 'outputs' is of shape [batch_size, timesteps, num_units]
            outputs, _ = tf.nn.dynamic_rnn(network,
                                           inputs_rnn,
                                           sequence_length=reduced_lens,
                                           dtype=tf.float32)

            # Extract the last relevant output from dynamic run using variable-length sequences
            last = self._extract_relevant(outputs, reduced_lens - 1)

        elif self._direction == 'bi':
            # Bidirectional
            cell_fw = rnn_cell(self._num_units, self.keep_prob, rnn_type=self._rnn_type)
            cell_bw = rnn_cell(self._num_units, self.keep_prob, rnn_type=self._rnn_type)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                         cell_bw,
                                                         inputs_rnn,
                                                         sequence_length=reduced_lens,
                                                         dtype=tf.float32)
            # Unpack forward/backward outputs
            output_fw, output_bw = outputs

            # Not sure why cannot use the staticmethod here!
            # last_fw = self._extract_relevant(output_fw, self.seq_lens - 1)
            batch_range = tf.range(tf.shape(output_fw)[0])
            indices = tf.stack([batch_range, reduced_lens - 1], axis=1)
            last_fw = tf.gather_nd(output_fw, indices)

            # The last output from the backward layer locates at the first timestep
            last_bw = output_bw[:, 0, :]
            last = tf.concat([last_fw, last_bw], 1)

        else:
            raise ValueError("Unknown specified direction '{}'".format(self._direction))

        # Apply ReLu
        relu = tf.nn.relu(last, name='relu')

        # Apply dropout
        drop = tf.nn.dropout(relu, self.keep_prob)

        # For binary classification, we just need one unit from the fully-connected
        # Weights are initialized using truncated normal initializer
        # Biases are initialized with non-zero for layer pre-activation
        self.logits = tf.layers.dense(drop,
                                      units=Config.num_classes,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name='dense')
        # self.predictions = tf.cast(tf.greater(self.logits, 0.), tf.float32)
        self.predictions = tf.nn.softmax(self.logits)


def run_experiment(model_, data, config, log_file):
    """
    Run experiment with data and setup

    Args:
        model_:   Model class pointer
        data:     Contains train/test inputs, targets, sequence lengths
        config:   Configuration parameters
        log_file: Log filename

    Returns:
        records: Contains train/test results

    """
    log_dir = '/resources/data/polygraph/{}.txt'.format(log_file)
    header = "direction: {}, num_units: {}, num_layers: {}, keep_prob: {}, lr: {}\n"
    with open(log_dir, 'w') as logger:
        logger.write(header.format(config.direction,
                                   config.num_units,
                                   config.num_layers,
                                   config.keep_prob,
                                   config.lr))
    # Unpack data for train/test
    X_train, X_test, y_train, y_test, l_train, l_test = data

    train_size = X_train.shape[0]

    records = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': [],
               'test_auc': []}

    tf.reset_default_graph()

    with tf.Graph().as_default():
        # Construct TF graph
        model = model_(config.num_units,
                       config.num_layers,
                       rnn_type=config.rnn_type,
                       direction=config.direction,
                       lr=config.lr)
        model.build_graph()

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            batch_log = "\tBatch {}: F/T {}, batch_acc {:.1%}"
            train_log = "Epoch {}/{}: train_loss {:.4f}, train_acc {:.1%}," \
                        "test_loss {:.4f} test_acc {:.1%}, test_auc {:.1%}"

            for epoch in range(1, config.num_epochs + 1):
                _loss, _acc = 0, 0
                # Generate training batches
                X_batches, y_batches, l_batches = make_batches(X_train,
                                                               y_train,
                                                               l_train,
                                                               batch_size=config.batch_size)
                print("{}/{}:".format(epoch, config.num_epochs))
                for i, (X_batch, y_batch, l_batch) in enumerate(zip(X_batches, y_batches, l_batches), start=1):
                    # Number of samples in the current batch
                    num_samples_batch = len(l_batch)
                    class_batch = y_batch.sum(axis=0)

                    # pos_weight = num_F/num_T, a weight used on the positive class
                    num_neg_batch, num_pos_batch = class_batch
                    batch_loss, batch_acc, _ = model.run(session,
                                                         X_batch, y_batch, l_batch,
                                                         config.keep_prob,
                                                         pos_weight=num_neg_batch / num_pos_batch)

                    _acc += batch_acc * num_samples_batch
                    _loss += batch_loss * num_samples_batch

                    if i % 10 == 0:
                        print(batch_log.format(i, class_batch, batch_acc))

                train_loss = _loss / train_size
                train_acc = _acc / train_size
                records['train_loss'].append(train_loss)
                records['train_acc'].append(train_acc)

                # Evaluation
                test_loss, test_acc, y_pred = model.run(session, X_test, y_test, l_test, 1.0, train=False)
                test_auc = roc_auc_score(y_test, y_pred)
                records['test_loss'].append(test_loss)
                records['test_acc'].append(test_acc)
                records['test_auc'].append(test_auc)

                log = train_log.format(epoch, config.num_epochs,
                                       train_loss, train_acc,
                                       test_loss, test_acc, test_auc)
                print(log)
                with open(log_dir, 'a') as logger:
                    logger.write(log + "\n")

    return records
