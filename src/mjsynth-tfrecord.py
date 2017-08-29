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
import math
#from pudb import set_trace; set_trace()

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  image/text: string specifying the human-readable version of the text
"""

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="abcdefghijklmnopqrstuvwxyz0123456789./-"

jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=1)

kernel_sizes = [5,5,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1024)]

def gen_data(input_base_dir, image_list_filename, output_filebase, num_shards=3, start_shard=0):
    """ Generate several shards worth of TFRecord data """
    count_skip = 0
    count_done = 0
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    image_filenames = get_image_filenames(os.path.join(input_base_dir, image_list_filename))
    num_digits = math.ceil( math.log10( num_shards - 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    
    for i in range(start_shard,num_shards):
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        if os.path.isfile(out_filename): # Don't recreate data if restarting
            continue
        #print(str(i),'of',str(num_shards),'[',str(start),':',str(end),']',out_filename)
        count_skip, count_done = gen_shard(sess, input_base_dir, image_filenames[start:end], out_filename, count_skip, count_done)
    # Clean up writing last shard
    start = num_shards*images_per_shard
    out_filename = output_filebase+'-'+(shard_format % num_shards)+'.tfrecord'
    #print(str(i),'of', str(num_shards), '[',str(start),':]', out_filename)
    count_skip, count_done = gen_shard(sess, input_base_dir, image_filenames[start:], out_filename, count_skip, count_done)

    print('исключено изображений:', count_skip)
    print('занесено изображений:', count_done)
    sess.close()

def gen_shard(sess, input_base_dir, image_filenames, output_filename, count_skip, count_done):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for filename in image_filenames:
        file_dir = filename[1 : filename.find(' ')]
        path_filename = os.path.join(input_base_dir, file_dir)
        if os.stat(path_filename).st_size == 0:
            #нулевой размер файла
            count_skip += 1
            #print('SKIPPING', file_dir)
            continue
        try:
            image_data, height, width = get_image(sess, path_filename)
            text = filename[filename.find(' ')+1 : ]
            labels = get_text_and_labels(text)
            if is_writable(width, height, text):
                count_done += 1
                example = make_example(file_dir, image_data, labels, text, height, width)
                writer.write(example.SerializeToString())
            else:
                #количество символов на изображении больше, чем ширина изображения после CNN слоев
                count_skip += 1
                #print('SKIPPING', file_dir)
        except:
            # Some files have bogus payloads, catch and note the error, moving on
            count_skip += 1
            #print('ERROR', file_dir)
    writer.close()
    return count_skip, count_done


def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            line = line.strip()
            filenames.append(line)
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f: # если просто 'r', то возникает ошибка UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
        image_data = f.read()
    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

def is_writable(image_width, image_height, text):
    """Determine whether the CNN-processed image is longer than the string"""
    writable = False

    if image_width > min_width:
        if len(text) <= seq_lens[image_width]:
            writable = True
    
    if len(text) > 2:
        if image_width/image_height < 1:
            writable = False
    
    return writable
    
def get_text_and_labels(text):
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    labels = [out_charset.index(c) for c in list(text)]
    return labels

def make_example(filename, image_data, labels, text, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_data: string, JPEG encoding of grayscale image
    labels: integer list, identifiers for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def main(argv=None):
    gen_data('/Users/kalinin/Desktop/Dataset', 'dir_train.txt', '/Users/kalinin/Desktop/Dataset/tfrecords/train', num_shards=5)
    gen_data('/Users/kalinin/Desktop/Dataset', 'dir_test.txt', '/Users/kalinin/Desktop/Dataset/tfrecords/test', num_shards=2)
    gen_data('/Users/kalinin/Desktop/Dataset', 'dir_val.txt', '/Users/kalinin/Desktop/Dataset/tfrecords/val', num_shards=2)

if __name__ == '__main__':
    main()
