import os
import tensorflow as tf
from generate import WGan_GP
from dataset import Dataset


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        if keys=="log_dir":
            
            FLAGS.__delattr__(keys)
del_all_flags(tf.compat.v1.flags.FLAGS)
flags = tf.compat.v1.app.flags


flags.DEFINE_float("learn_rate", 0.0002, "the learning rate for gan")
# flags.DEFINE_integer("output_size", 28 , "the size of generate image")
flags.DEFINE_integer("batch_size", 512, "the batch number")
flags.DEFINE_integer("z_dim", 16, "the dimension of noise z")
flags.DEFINE_integer("c_dim", 4, "the dimension of label c")
flags.DEFINE_integer("seq_length", 32, "the length of music sequence")
flags.DEFINE_integer("note_dim", 9, "the dimension of single note")
flags.DEFINE_string("sample_dir" , "sample" , "the dir of sample images")
flags.DEFINE_string("log_dir" , "log" , "the path of tensorflow's log")
flags.DEFINE_string("model_dir" , "model" , "the read and save path of model")
flags.DEFINE_string("model_id" , "wgan_gp_v1.0" , "the read and save path of model")

FLAGS = flags.FLAGS
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)


def main(_):

    dataset = Dataset(FLAGS.batch_size)
    model = WGan_GP(data = dataset, output_size=(16,27), learn_rate=FLAGS.learn_rate, batch_size=FLAGS.batch_size
    , z_dim=FLAGS.z_dim, c_dim=FLAGS.c_dim, sample_dir = FLAGS.sample_dir, log_dir=FLAGS.log_dir , model_dir=FLAGS.model_dir, seq_length=FLAGS.seq_length, note_dim=FLAGS.note_dim, model_id=FLAGS.model_id)
    model.build_model()
    model.train()

if __name__ == '__main__':
    tf.compat.v1.app.run()
