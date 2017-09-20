# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import mjsynth
import model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output','../data/model',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size',2**5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',2**16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_float('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")


tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")

tf.app.flags.DEFINE_string('train_device','/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device','/gpu:0',
                           """Device for preprocess/batching graph placement""")

tf.app.flags.DEFINE_string('train_path','../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern','words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',1, # выборка будет собираться из нескольких tfrecords (using threads=N will create N copies of the reader op connected to the queue so that they can run in parallel)
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")

tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'
mode = learn.ModeKeys.TRAIN # 'Configure' training mode for dropout layers

def _get_input():
    """Set up and return image, label, and image width tensors"""

    image, width, label, length, text, filename, number_of_images = mjsynth.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold )

    #tf.summary.image('images',image) # Uncomment to see images in TensorBoard
    return image, width, label, length, text, filename, number_of_images

def _get_single_input():
    """Set up and return image, label, and width tensors"""

    image,width,label,length,text,filename=mjsynth.threaded_input_pipeline(
        deps.get('records'), 
        str.split(FLAGS.filename_pattern,','),
        batch_size=1,
        num_threads=FLAGS.num_input_threads,
        num_epochs=1,
        batch_device=FLAGS.input_device, 
        preprocess_device=FLAGS.input_device )
    return image,width,label,length,text,filename

def _get_training(rnn_logits, label, sequence_length, label_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)

        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum)
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate, 
                optimizer=optimizer,
                variables=rnn_vars)

            tf.summary.scalar( 'learning_rate', learning_rate )

    with tf.name_scope("test"):
        predictions, probability = tf.nn.ctc_beam_search_decoder(rnn_logits, # вытаскивание log_probabilities из ctc
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=False) # если True, то на выходе модели не будет повторяющихся символов
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False) # расстояние Левенштейна
        sequence_errors = tf.count_nonzero(label_errors, axis=0)
        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error, tf.cast(total_labels, tf.float32), name='label_error')
        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32), tf.shape(label_length)[0], name='sequence_error')
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('label_error', label_error)
        tf.summary.scalar('sequence_error', sequence_error)

    return train_op, label_error, sequence_error, predictions[0], probability

def _get_session_config():
    """Setup session config to soften device placement"""

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None
    
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ckpt_path=FLAGS.tune_from

    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def main(argv=None):

    with tf.Graph().as_default(): # формальная (если граф в программе всего один) конструкция для объединения операция в отдельный граф, https://stackoverflow.com/questions/39614938/why-do-we-need-tensorflow-tf-graph , https://danijar.com/what-is-a-tensorflow-session/
        tf.set_random_seed(1) # фиксация сида, на gpu в вычислениях писутствует случайная составляющся и результаты все равно могут немного отличаться
        global_step = tf.contrib.framework.get_or_create_global_step() # переменная для подсчета количество эпох (?)
        
        image, width, label, length, text, filename, number_of_images = _get_input() # формирование выборки для обучения

        with tf.device(FLAGS.train_device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            train_op, label_error, sequence_error, predict, prob = _get_training(logits, label, sequence_length, length)

        session_config = _get_session_config()

        summary_op = tf.summary.merge_all() # Merges all summaries collected in the default graph.
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 
        step_ops = [global_step, train_op, label_error, sequence_error, tf.sparse_tensor_to_dense(label), tf.sparse_tensor_to_dense(predict), text, filename, prob]
        sv = tf.train.Supervisor(
            logdir=FLAGS.output,
            init_op=init_op,
            summary_op=summary_op,
            save_summaries_secs=30, # Number of seconds between the computation of summaries for the event log
            init_fn=_get_init_pretrained(), # None если train запускается без предобученных весов
            save_model_secs=150) # Number of seconds between the creation of model checkpoints

        try:
            loss_change = np.load('./train_loss.npy').item().get('loss_change')
            accuracy_change = np.load('./train_loss.npy').item().get('accuracy_change')
            Levenshtein_change = np.load('./train_loss.npy').item().get('Levenshtein_change')
            Levenshtein_nonzero_change = np.load('./train_loss.npy').item().get('Levenshtein_nonzero_change')
            print('metrics and loss are loaded')
        except:
            loss_change = []
            accuracy_change = []
            Levenshtein_change = []
            Levenshtein_nonzero_change = []
            print('metrics and loss are created')
        with sv.managed_session(config=session_config) as sess:
            step = sess.run(global_step)
            while step < FLAGS.max_num_steps:
                if sv.should_stop():
                    break

                step_vals = sess.run(step_ops)

                accuracy = 0
                out_charset = "abcdefghijklmnopqrstuvwxyz0123456789./-"
                for pred in range(len(step_vals[5])):
                    pred_txt = ''
                    for symb in step_vals[5][pred].tolist():
                        pred_txt += str(out_charset[symb])
                    pred_txt_clear = ''
                    stop_pass = False
                    # в выходе модели пустые символы в конце стркои заполняются первым символом out_charset
                    for symb in pred_txt[::-1]:
                        if symb == out_charset[0] and stop_pass == False:
                            pass
                        else:
                            pred_txt_clear = symb + pred_txt_clear
                            stop_pass = True
                    if pred_txt_clear == step_vals[6][pred].decode('utf-8'):
                        accuracy += 1

                # print(step_ops[7]) # вывод на экран собранного батча для обучения
                loss_change.append(step_vals[1])
                accuracy_change.append(accuracy/len(step_vals[5]))
                Levenshtein_change.append(step_vals[2])
                Levenshtein_nonzero_change.append(step_vals[3])

                if step_vals[0]%100==0:
                    print('loss', np.mean(loss_change[-100:]))
                    print('sum Levenshtein on the batch', sum(Levenshtein_change[-100:]))
                    print('sum Levenshtein nonzero', sum(Levenshtein_nonzero_change[-100:]))
                    print('accuracy', np.mean(accuracy_change[-100:]))
                    # сохранение лосса и других статистик
                    np.save('./train_loss', {
                        'loss_change':loss_change,
                        'accuracy_change':accuracy_change,
                        'Levenshtein_change':Levenshtein_change,
                        'Levenshtein_nonzero_change':Levenshtein_nonzero_change
                        })
            sv.saver.save( sess, os.path.join(FLAGS.output,'model.ckpt'),
                           global_step=step_vals[0])

if __name__ == '__main__':
    tf.app.run()
