import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import mjsynth
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output', '../data/model', 'Directory for event logs and checkpoints')
tf.app.flags.DEFINE_string('tune_from', '', 'Path to pre-trained model checkpoint')
tf.app.flags.DEFINE_string('tune_scope', '', 'Variable scope for training')
tf.app.flags.DEFINE_integer('save_and_print_frequency', 100, 'Save and print frequency')

tf.app.flags.DEFINE_integer('batch_size', 1, 'Mini-batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Optimizer gradient first-order momentum')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Learning rate decay base')
tf.app.flags.DEFINE_float('decay_steps', 2**16, 'Learning rate decay exponent scale')
tf.app.flags.DEFINE_float('decay_staircase', False, 'Staircase learning rate decay by integer division')
tf.app.flags.DEFINE_integer('max_num_steps', 2**21, 'Number of optimization steps to run')

tf.app.flags.DEFINE_string('train_device', '/gpu:1', 'Device for training graph placement')
tf.app.flags.DEFINE_string('input_device', '/gpu:0', 'Device for preprocess/batching graph placement')

tf.app.flags.DEFINE_string('train_path', '../data/train/', 'Base directory for training data')
tf.app.flags.DEFINE_string('filename_pattern', 'words-*', 'File pattern for input data')
tf.app.flags.DEFINE_integer('num_input_threads', 1, 'Number of readers for input data') # выборка будет собираться из нескольких tfrecords (using threads=N will create N copies of the reader op connected to the queue so that they can run in parallel)
tf.app.flags.DEFINE_integer('width_threshold', None, 'Limit of input image width')
tf.app.flags.DEFINE_integer('length_threshold', None, 'Limit of input string length width')

tf.logging.set_verbosity(tf.logging.INFO)

mode = learn.ModeKeys.TRAIN # 'Configure' training mode for dropout layers

def _get_input():
    # Set up and return image, label, and image width tensors
    image, width, label, length, text, filename, number_of_images = mjsynth.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold)
    return image, width, label, length, text, filename, number_of_images

def _get_training(rnn_logits, label, sequence_length, label_length):
    # Set up training ops
    with tf.name_scope('train'):
        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope='convnet|rnn'
        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        loss = model.ctc_loss_layer(rnn_logits, label, sequence_length)
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

    with tf.name_scope('test'):
        predictions, probability = tf.nn.ctc_beam_search_decoder(rnn_logits, # вытаскивание log_probabilities из ctc
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=False) # если True, то на выходе модели не будет повторяющихся символов
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False) # расстояние Левенштейна
        sequence_errors = tf.count_nonzero(label_errors, axis=0) # подсчет ненулевых значнией, то есть случаев, когда выход модели не совпадает с gt
        total_label_error = tf.reduce_sum(label_errors) # рассчет суммы расстояний Левенштейна по батчу
        total_labels = tf.reduce_sum(label_length) # количество символов в gt для всего батча
        label_error = tf.truediv(total_label_error, tf.cast(total_labels, tf.float32), name='label_error') # нормированное расстояние Левенштейна (деленное на количество символов)
        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32), tf.shape(label_length)[0], name='sequence_error') # доля неправильных ответов

    return train_op, label_error, sequence_error, predictions[0], probability

def _get_session_config():
    # Setup session config to soften device placement
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True, 
        log_device_placement=False)
    return config

def main(argv=None):

    with tf.Graph().as_default(): # формальная (если граф в программе всего один) конструкция для объединения операция в отдельный граф, https://stackoverflow.com/questions/39614938/why-do-we-need-tensorflow-tf-graph , https://danijar.com/what-is-a-tensorflow-session/
        tf.set_random_seed(1) # фиксация сида, на gpu в вычислениях писутствует случайная составляющся и результаты все равно могут немного отличаться
        global_step = tf.contrib.framework.get_or_create_global_step() # переменная для подсчета количество эпох
        
        image, width, label, length, text, filename, number_of_images = _get_input() # формирование выборки для обучения
        
        with tf.device(FLAGS.train_device):
            features,sequence_length = model.convnet_layers(image, width, mode)
            logits = model.rnn_layers(features, sequence_length, mjsynth.num_classes())
            train_op, label_error, sequence_error, predict, prob = _get_training(logits, label, sequence_length, length)

        session_config = _get_session_config()
        
        saver_reader = tf.train.Saver(max_to_keep=100)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
        
        step_ops = [global_step, train_op, label_error, sequence_error, tf.sparse_tensor_to_dense(label), tf.sparse_tensor_to_dense(predict), text, filename, prob]
        
        try:
            loss_change = np.load('./train_loss.npy').item().get('loss_change')
            Levenshtein_change = np.load('./train_loss.npy').item().get('Levenshtein_change')
            accuracy_change = np.load('./train_loss.npy').item().get('accuracy_change')
            print('metrics and loss are loaded')
        except:
            loss_change = []
            Levenshtein_change = []
            accuracy_change = []
            print('metrics and loss are created')
        with tf.Session(config=session_config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator() # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # подгрузка весов
            if os.path.isdir(FLAGS.output):
                ckpt = tf.train.get_checkpoint_state(FLAGS.output)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_path = ckpt.model_checkpoint_path
                else:
                    raise RuntimeError('No checkpoint file found')
                # init_fn = lambda sess, ckpt_path: saver_reader.restore(sess, ckpt_path)
                saver_reader.restore(sess, ckpt_path)
            step = sess.run(global_step)
            while step < FLAGS.max_num_steps:

                step_vals = sess.run(step_ops)
                step = step_vals[0]

                out_charset = 'abcdefghijklmnopqrstuvwxyz0123456789./-'
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

                # print(step_ops[7]) # вывод на экран собранного батча для обучения
                loss_change.append(step_vals[1])
                Levenshtein_change.append(step_vals[2])
                accuracy_change.append(1-step_vals[3])
                print(step_vals[1])
                if step_vals[0]%FLAGS.save_and_print_frequency==0:
                    print('loss', np.mean(loss_change[-FLAGS.save_and_print_frequency:]))
                    print('mean Levenshtein', np.mean(Levenshtein_change[-FLAGS.save_and_print_frequency:]))
                    print('accuracy', np.mean(accuracy_change[-FLAGS.save_and_print_frequency:]))
                    # сохранение лосса и других статистик
                    np.save('./train_loss', {
                        'loss_change':loss_change,
                        'Levenshtein_change':Levenshtein_change,
                        'accuracy_change':accuracy_change})
                    saver_reader.save(sess, os.path.join(FLAGS.output, 'model.ckpt'), global_step=step)
        coord.join(threads)

if __name__ == '__main__':
    # from pudb import set_trace; set_trace()
    tf.app.run()