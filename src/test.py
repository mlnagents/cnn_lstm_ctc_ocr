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
import time
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import mjsynth
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('output','test',
                          """Sub-directory of model for test summary events""")

tf.app.flags.DEFINE_integer('batch_size',2**8,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 0, # зачем-то time.sleep в конце кода перед следующей итерацией теста
                             'Time between test runs')

tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")

tf.app.flags.DEFINE_string('test_path','../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern','test/words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',1,
                          """Number of readers for input data""")

tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers


def _get_input():
    """Set up and return image, label, width and text tensors"""

    image, width, label, length, text, filename, number_of_images = mjsynth.bucketed_input_pipeline(
        FLAGS.test_path,
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold)
    
    return image, width, label, length, text, filename, number_of_images

def _get_session_config():
    """Setup session config to soften device placement"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False,
        gpu_options=gpu_options)

    return config

def _get_testing(rnn_logits,sequence_length,label,label_length):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    with tf.name_scope("train"):
        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 
    with tf.name_scope("test"):
        predictions, probability = tf.nn.ctc_beam_search_decoder(rnn_logits, # вытаскивание log_probabilities из ctc
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=False) # если True, то на выходе модели не будет повторяющихся символов
        #predictions, probability = tf.nn.ctc_greedy_decoder(rnn_logits,  # альтернативный loss (?)
        #                                           sequence_length,
        #                                           merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False) # расстояние Левенштейна
        sequence_errors = tf.count_nonzero(label_errors,axis=0)
        total_label_error = tf.reduce_sum( label_errors )
        total_labels = tf.reduce_sum( label_length )
        label_error = tf.truediv( total_label_error, 
                                  tf.cast(total_labels, tf.float32 ),
                                  name='label_error')
        sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
                                     tf.shape(label_length)[0],
                                     name='sequence_error')
        tf.summary.scalar( 'loss', loss )
        tf.summary.scalar( 'label_error', label_error )
        tf.summary.scalar( 'sequence_error', sequence_error )

    return loss, label_error, sequence_error, predictions[0], probability

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


def main(argv=None):

    with tf.Graph().as_default():
        image, width, label, length, text, filename, number_of_images = _get_input() # извлечение выборки изображений для теста
        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            loss,label_error,sequence_error, predict, prob = _get_testing(
                logits,sequence_length,label,length)

        global_step = tf.contrib.framework.get_or_create_global_step()

        session_config = _get_session_config()
        restore_model = _get_init_trained()
        
        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        summary_writer = tf.summary.FileWriter( os.path.join(FLAGS.model,
                                                            FLAGS.output) )

        step_ops = [global_step, loss, label_error, sequence_error, tf.sparse_tensor_to_dense(label), tf.sparse_tensor_to_dense(predict), text, filename, prob, logits]

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            count_list = []
            enlisted_images = 0
            coord = tf.train.Coordinator() # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            summary_writer.add_graph(sess.graph)
            loss_change = []
            accuracy_change = []
            Levenshtein_change = []
            Levenshtein_nonzero_change = []

            try:            
                while True:
                    restore_model(sess, _get_checkpoint()) # Get latest checkpoint

                    if not coord.should_stop():
                        step_vals = sess.run(step_ops)
                        # print(step_vals[7]) # вывод на экран батча
                        # out_charset = "abcdefghijklmnopqrstuvwxyz0123456789./-%"
                        # a = step_vals[-1]
                        # a = np.reshape(a, (step_vals[-1].shape[0],40))
                        # max_a = 0
                        # max_a_j = 0
                        # sum_a = 0
                        # total_mult = 0
                        # total_add = 0
                        # with open('./result.txt', 'a') as f:
                        #     for q in range(step_vals[-1].shape[0]):
                        #         for j in range(40):
                        #             if a[q][j] >= 0:
                        #                 sum_a += np.exp(a[q][j])
                        #             if a[q][j] > max_a:
                        #                 max_a = a[q][j]
                        #                 max_a_j = j
                        #         if str(out_charset[max_a_j]) != '%' and sum_a > 0:
                        #             f.write(str(max_a) + ' = ' + str(out_charset[max_a_j]) +' (' + str(round(np.log(np.exp(max_a)/sum_a), 5)) +') ' + '\n')
                        #             total_mult = total_mult*np.log(np.exp(max_a)/sum_a)
                        #             total_add += np.log(np.exp(max_a)/sum_a)
                        #         max_a = 0
                        #         max_a_j = 0
                        #         sum_a = 0
                        #     f.write(str(total_mult) + '\n')
                        #     f.write(str(total_add) + '\n')
                        with open('./result.txt', 'a') as f:
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
                                # в итоговом txt файле сохраняются только уникальные изображения
                                # батчи формируются последовательно без повторений
                                # однако последний тестовый батч может содержать часть изображений первого батча
                                if step_vals[7][pred] not in count_list:
                                    enlisted_images += 1
                                    f.write(pred_txt_clear + ' ') # прогноз модели
                                    f.write(step_vals[6][pred].decode('utf-8') + ' ') # ground truth
                                    f.write(str(step_vals[8][pred][0]) + ' ') # вероятности
                                    f.write(step_vals[7][pred].decode('utf-8') + '\n') # директории
                                    count_list.append(step_vals[7][pred])
                                    # остановить тест, когда количество уникальных изображений равно или больше количества изображений в датасете
                                    if enlisted_images >= number_of_images:
                                        coord.request_stop()

                        print(round(enlisted_images/number_of_images, 2)) # процент пройденного датасета
                        loss_change.append(step_vals[1])
                        accuracy_change.append(accuracy/len(step_vals[5]))
                        Levenshtein_change.append(step_vals[2])
                        Levenshtein_nonzero_change.append(step_vals[3])
                        print('Batch done')
                        # summary_str = sess.run(summary_op) # вызывает повторное извлечение батча, который не используется моделью
                        # summary_writer.add_summary(summary_str,step_vals[0])
                    else:
                        print('loss', np.mean(loss_change))
                        print('sum Levenshtein on the batch', np.mean(Levenshtein_change))
                        print('sum Levenshtein nonzero', np.mean(Levenshtein_nonzero_change))
                        print('accuracy', np.mean(accuracy_change))
                        print('Test done')
                        break
                    time.sleep(FLAGS.test_interval_secs)
            except tf.errors.OutOfRangeError:
                    print('Done')
            finally:
                coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # from pudb import set_trace; set_trace()
    tf.app.run()
