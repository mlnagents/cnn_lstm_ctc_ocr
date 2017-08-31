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
tf.app.flags.DEFINE_string('filename_pattern','val/words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers


def _get_input():
    """Set up and return image, label, width and text tensors"""

    image,width,label,length,text,filename=mjsynth.threaded_input_pipeline(
        FLAGS.test_path,
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        num_epochs=None, # Repeat for streaming
        batch_device=FLAGS.device, 
        preprocess_device=FLAGS.device )
    
    return image,width,label,length,text,filename

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
        label_errors = tf.edit_distance(hypothesis, label, normalize=False)
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
        image,width,label,length,text,filename = _get_input()

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

        step_ops = [global_step, loss, label_error, sequence_error, tf.sparse_tensor_to_dense(label), tf.sparse_tensor_to_dense(predict), text, filename, prob]

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            count_list = []

            coord = tf.train.Coordinator() # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            
            summary_writer.add_graph(sess.graph)

            try:            
                while True:
                    restore_model(sess, _get_checkpoint()) # Get latest checkpoint

                    if not coord.should_stop():
                        step_vals = sess.run(step_ops)
                        print(step_vals[0:4])
                        print('Done')
                        with open('/cnn_lstm_ctc_ocr/src/result.txt', 'a') as f:
                            out_charset = "abcdefghijklmnopqrstuvwxyz0123456789./-"
                            for pred in range(len(step_vals[5])):
                                pred_txt = ''
                                for symb in step_vals[5][pred].tolist():
                                    pred_txt += str(out_charset[symb])
                                pred_txt_clear = ''
                                stop_pass = False
                                for symb in pred_txt[::-1]:
                                    if symb == 'a' and stop_pass == False:
                                        pass
                                    else:
                                        pred_txt_clear = symb + pred_txt_clear
                                        stop_pass = True
                                if step_vals[7][pred] not in count_list: # временное решения для сохранения в результатах теста только уникальных изображений (для теста изображения извлекаются случайно с повтором)
                                    f.write(pred_txt_clear + ' ')
                                    f.write(step_vals[6][pred].decode('utf-8') + ' ')
                                    f.write(str(step_vals[8][pred][0]) + ' ')
                                    f.write(step_vals[7][pred].decode('utf-8') + '\n')
                                    count_list.append(step_vals[7][pred])

                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str,step_vals[0])
                    else:
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
