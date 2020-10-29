# from utils import save_images, vis_square,sample_label
# from tensorflow.contrib.layers.python.layers import xavier_initializer
import tensorflow as tf
import numpy as np
import data
import os
from tensorflow import keras
from tensorflow.keras import layers,Input,optimizers
from tensorflow.keras.models import Model
from midi_io import midi_file_to_note_sequence,note_sequence_to_midi_file
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, enable=True)


class Vitalize(object):

    # build model
    def __init__(self, data, output_size, learn_rate, batch_size, z_dim, c_dim, sample_dir, log_dir, model_dir, seq_length, note_dim, model_id):

        self.data = data
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.note_dim = note_dim
        self.sample_dir = './'+sample_dir+'/'+model_id
        self.log_dir = './'+log_dir+'/'+model_id
        self.model_dir = './'+model_dir+'/'+model_id
        self.seq_length = seq_length
        self.model_id = model_id
        self.check_path()

    def build_model(self):

        label_size=self.c_dim
        embedding_dim=9
        optimizer=optimizers.Adam(0.0003)

        hits=Input(shape=(None,9),dtype='float32',name='musics')
        style=Input(shape=(None,),dtype='int8',name='style')

        encode_style=layers.Embedding(label_size+1,embedding_dim)(style)
        concatenated=layers.concatenate([encode_style,hits])

        latent=layers.Bidirectional(layers.LSTM(16,return_sequences=True,dropout=0.4))(concatenated)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(32,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(64,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(128,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])

        output=layers.Dense(18,activation='tanh')(latent)

        self.model=Model([hits,style],output)
        self.model.compile(loss=['MeanSquaredError'],optimizer=optimizer)


    def train(self, sample_interval=30,save_interval=50):

        writer=tf.summary.create_file_writer(self.log_dir)
        # self.load_model(self.model_dir)
        with writer.as_default():
            epochs=2000*self.data.length//self.batch_size
            train_steps=20
            test_steps=5
            for step in range(epochs):
                for train_step in range(train_steps):
                    x,y,label=self.data.next_pretrain_batch(step*train_steps+train_step)
                    label=tf.repeat(label.reshape(self.batch_size,1),repeats=self.seq_length,axis=1)
                    train_loss=self.model.train_on_batch([x,label],y)
                    tf.summary.scalar("train_loss",train_loss,step=step*train_steps+train_step)
                    print('step %d/%d/%d train_loss %.4f'%(step,epochs,train_step,train_loss))
                    writer.flush()
                for test_step in range(test_steps):
                    x,y,label=self.data.next_pretrain_batch(step*test_steps+test_step)
                    label=tf.repeat(label.reshape(self.batch_size,1),repeats=self.seq_length,axis=1)
                    test_loss=self.model.train_on_batch([x,label],y)
                    tf.summary.scalar("test_loss",test_loss,step=step*test_steps+test_step)
                    print('step %d/%d/%d test_loss %.4f'%(step,epochs,test_step,test_loss))
                    writer.flush()
                
                if step % (save_interval*(self.data.length//self.batch_size)) == 0 and step!=0:
                    self.model.save(self.model_dir+'/model_step_%03dk'%(step/1000))
                
    def check_path(self):
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


