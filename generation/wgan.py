
import tensorflow as tf
import numpy as np
import data
import os,sys
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import Input,layers,optimizers
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from midi_io import midi_file_to_note_sequence,note_sequence_to_midi_file
matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, enable=True)
tf.compat.v1.disable_eager_execution()
class WGan_GP:
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
        self.n_discriminator = 5
        self.check_path()

    def build_model(self):
        print('create the generator unit')
        
        self.gen_model = Generator(self.c_dim,self.z_dim).modelling()
        # self.gen_model.summary()
        print('create the discriminator unit')
        
        self.discrim_model = Discriminator(self.c_dim,self.seq_length,self.note_dim,self.z_dim).modelling()
        # self.discrim_model.summary()

        print('create the combined model(update discriminator)')


        discriminator = Discriminator(self.c_dim, self.seq_length,self.note_dim, self.z_dim,self.gen_model, self.discrim_model)
        self.discriminator_train_func = discriminator.build()
        
        print('create the combined model(update generator)')
        combined = Generator(self.c_dim,self.z_dim,self.gen_model, self.discrim_model)
        self.generator = combined.generator
        self.combined_train_func = combined.build()
        
        print('create the vitalize model')
        self.vitalize = load_model('./model/model_step_026k')


    def check_path(self):
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_model(self,path):
        self.discrim_model = load_model(self.model_dir+'/discriminator')
        self.gen_model = load_model(self.model_dir+'/discriminator')

    def save_model(self,step,path):
        self.generator.save(path+'/generator_step_%03dk'%(step/1000))

    def generate_fake_musics(self,seq_length):
        hits = np.random.normal(0, 1, (self.batch_size,seq_length,9))
        hits = np.where(hits>0.8,1,0)
        offset = np.random.normal(0, 1, (self.batch_size,seq_length,9))
        offset = np.where(offset>0.5,0.5,offset)
        offset = np.where(offset<-0.5,-0.5,offset)
        offset = np.multiply(offset,hits)
        velocity = np.random.normal(0, 1, (self.batch_size,seq_length,9))
        velocity = np.where(velocity>0.5,0.5,velocity)
        velocity = np.where(velocity<-0.5,-0.5,velocity)
        velocity = np.multiply(velocity,hits)
        return np.concatenate([hits,velocity,offset],axis=-1)

    def sample_musics(self,step,path):
        print('sample music....')
        music_length=32
        sample_num=20
        data_converter=data.GrooveConverter(
                split_bars=music_length//16,
                steps_per_quarter=4, quarters_per_bar=4,
                max_tensors_per_notesequence=50,
                pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
                inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES)
        noise = np.random.normal(0, 1, (sample_num, music_length, self.z_dim))
        fake_label = np.random.randint(1,self.c_dim+1,size=(sample_num,1))
        fake_label = np.repeat(fake_label,music_length,axis=1)

        output_hits=self.generator.predict([noise,fake_label])
        # output_hits, output_velocities, output_offsets = tf.split(output,3, axis=-1)
        output_hits = tf.where(output_hits>0.5,1.0,0.0)
        
        output=self.vitalize.predict([output_hits,fake_label],steps=1)
        output_velocities, output_offsets = tf.split(output,2, axis=-1)
        output_velocities=tf.multiply(output_velocities,output_hits)
        output_offsets=tf.multiply(output_offsets,output_hits)
        output=tf.concat([output_hits,output_velocities,output_offsets],axis=-1)

        output=K.eval(output)
        seq=data_converter._to_notesequences(tuple(output))
        fake_label=fake_label.astype('int8')
        for i,ns in enumerate(seq):
            note_sequence_to_midi_file(ns,path+"/step_%05d_%04dth_label_%d.mid"%(step,i,fake_label[i,0]))
        np.savetxt(path+'/step_%d_label.txt'%step,fake_label[:,0])
        print('sample complete~~')

    # Train the Models(G && D)
    def train(self,sample_interval=50):
        writer=tf.summary.create_file_writer(self.log_dir)
        g_loss_draw=[]
        d_loss_draw=[]
        with writer.as_default():
            epochs=500*self.data.length//self.batch_size
            batch_size=self.batch_size
            # X_train = data
            # Rescale -1 to 1
            #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            #X_train = np.expand_dims(X_train, axis=3)

            # Adversarial ground truths
            valid =-np.ones((batch_size, 1))
            fake = np.ones((batch_size, 1))
            dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
            for epoch in range(1, epochs+1):
                    # ---------------------
                #  Train Discriminator
                # ---------------------

                for _ in range(self.n_discriminator):
                    # Select a random batch of images
                    idx = np.random.randint(0, self.data.length, batch_size)
                    # imgs = X_train[idx]
                    real_hits=self.data.data_x[idx]
                    label=self.data.data_label[idx]
                    label_dis=label.reshape(batch_size,1)
                    label=tf.repeat(label_dis,repeats=self.seq_length,axis=1)
                    # Generate noise randomly
                    noise = np.random.normal(0, 1, (batch_size, self.seq_length,self.z_dim))
                    # Get the alpha
                    alpha = np.random.uniform(size=(batch_size, 1 ,1))
                    # Train the discriminator
                    d_loss_real, d_loss_fake = self.discriminator_train_func([real_hits,noise,label,label_dis,alpha])
                    d_loss = d_loss_real - d_loss_fake
                    d_loss_draw.append(d_loss)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.seq_length,self.z_dim))
                label = np.random.randint(1,self.c_dim+1,batch_size)
                label_dis = label.reshape((self.batch_size,1))
                label = tf.repeat(label_dis,repeats=self.seq_length,axis=1)
                # Train the generator (to have the discriminator label samples as valid)
                g_loss, = self.combined_train_func([noise,label,label_dis])

                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
                
                g_loss_draw.append(g_loss)
                tf.summary.scalar("d_loss",d_loss,step=epoch)
                tf.summary.scalar("g_loss",g_loss,step=epoch)
                plt.figure(figsize=(20,10))
                plt.subplot(1,2,1)
                plt.plot(list(range(len(d_loss_draw))),d_loss_draw)
                plt.title('d_loss')
                plt.subplot(1,2,2)
                plt.title('g_loss')
                plt.plot(list(range(len(g_loss_draw))),g_loss_draw)
                plt.savefig(self.log_dir+'/loss.jpg')
                plt.clf()
                plt.close()
                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_musics(epoch,self.sample_dir)
                    self.gen_model.save(self.model_dir+'/generator_step_%05d'%(epoch))
                    self.discrim_model.save(self.model_dir+'/discriminator_step_%05d'%(epoch/1000))

class Generator:
    def __init__(self, c_dim,z_dim, gen_model=None, discrim_model=None):
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.generator = gen_model
        self.discriminator = discrim_model
    
    def build(self):
        # Only Train generator
        self.discriminator.trainable = False
        self.generator.trainable = True

        noise=Input(shape=(None,self.z_dim),dtype='float32',name='musics')
        style=Input(shape=(None,),dtype='int8',name='style')
        style_dis=Input(shape=(1,),dtype='int8',name='style')
        # Get Fake image with noise
        output = self.generator([noise,style])
        # generator loss function
        loss = -K.mean(self.discriminator([output,style_dis]))
        
        # Build the train strategy
        training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(params=self.generator.trainable_weights, loss=loss)
        train_func = K.function([noise,style,style_dis], [loss], training_updates)

        return train_func

    # Build Generator Model
    def modelling(self):
        embedding_dim=8
        label_size=4
        noise=Input(shape=(32,16),dtype='float32',name='musics')
        style=Input(shape=(32,),dtype='int8',name='style')

        encode_style=layers.Embedding(label_size,embedding_dim)(style)
        # style_modify=layers.Reshape((latent.shape[1],latent.shape[2],embedding_dim))(tf.repeat(encode_style,repeats=latent.shape[1]*latent.shape[2],axis=1))
        concatenated=layers.concatenate([encode_style,noise])

        latent=layers.Bidirectional(layers.LSTM(16,return_sequences=True,dropout=0.4))(concatenated)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(32,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(64,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])
        latent=layers.Bidirectional(layers.LSTM(128,return_sequences=True,dropout=0.4))(latent)
        latent=layers.concatenate([encode_style,latent])

        output=layers.Dense(9,activation='sigmoid')(latent)

        model=Model([noise,style],output)
        return model

class Discriminator:
    def __init__(self, label_size,seq_length,note_dim, latent_dim, gen_model=None, discrim_model=None):
        self.label_size=label_size
        self.seq_length=seq_length
        self.note_dim=note_dim
        self.latent_dim=latent_dim

        self.generator = gen_model
        self.discriminator = discrim_model

    # Get the Interpolated, In the middle of Real and Fake
    def get_interpolated(self, real_hits, fake_hits):
        alpha = K.placeholder(shape=(None,1,1))
        interpolated_hits = Input(shape=(None,9), tensor=alpha*real_hits+ (1-alpha)*fake_hits)
        return interpolated_hits, alpha

    # Gradient Penalty Calculation
    def gradient_penalty_loss(self, real_hits, fake_hits, interpolated_hits,style_dis):
        loss_real = K.mean(self.discriminator([real_hits,style_dis]))
        loss_fake = K.mean(self.discriminator([fake_hits,style_dis]))

        grad_mixed = K.gradients(self.discriminator([interpolated_hits,style_dis]), [interpolated_hits])[0]
        gradients_sqr = K.square(grad_mixed)
        norm_grad_mixed = K.sqrt(K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))))
        grad_penalty = K.mean(K.square(norm_grad_mixed-1))

        loss = loss_fake - loss_real + 10 * grad_penalty

        return loss_real, loss_fake, loss

    def build(self):
        # Only train discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True

        # Image input (real sample)
        real_hits = Input(shape=(None,9))
        noise=Input(shape=(None,self.latent_dim),dtype='float32',name='musics')
        style=Input(shape=(None,),dtype='int8',name='style')
        style_dis=Input(shape=(1,),dtype='int8',name='style')
        # Noise input
        # noise = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_hits = self.generator([noise,style])

        # Construct weighted average between real and fake images
        interpolated_hits,alpha = self.get_interpolated(real_hits, fake_hits)

        # Get loss of Gradient Penalty
        loss_real, loss_fake, loss = self.gradient_penalty_loss(real_hits, fake_hits, interpolated_hits,style_dis)

        # Build the train strategy
        training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(params=self.discriminator.trainable_weights, loss=loss)
        discriminator_train = K.function([real_hits, noise,style,style_dis,alpha],
                                [loss_real, loss_fake],    
                                training_updates)

        return discriminator_train

    # Build Discriminator Model
    def modelling(self):
        label_size=self.label_size
        length=self.seq_length
        note_dim=self.note_dim
        embedding_dim=8
        musics=Input(shape=(length,note_dim),dtype='float32',name='musics')
        style=Input(shape=(1,),dtype='int8',name='style')

        latent=layers.Reshape((length,note_dim,1))(musics)
        encode_style=layers.Embedding(label_size+1,embedding_dim)(style)

        style_modify=layers.Reshape((latent.shape[1],latent.shape[2],embedding_dim))(tf.repeat(encode_style,repeats=latent.shape[1]*latent.shape[2],axis=1))
        latent=layers.concatenate([latent,style_modify],axis=-1)
        latent=layers.Conv2D(16,3,(2,1))(latent)
        # latent=layers.BatchNormalization()(latent)
        latent=layers.LeakyReLU(alpha=0.2)(latent)
        latent=layers.Dropout(0.25)(latent)

        style_modify=layers.Reshape((latent.shape[1],latent.shape[2],embedding_dim))(tf.repeat(encode_style,repeats=latent.shape[1]*latent.shape[2],axis=1))
        latent=layers.concatenate([latent,style_modify],axis=-1)
        latent=layers.Conv2D(32,3,(2,1))(latent)
        # latent=layers.BatchNormalization()(latent)
        latent=layers.LeakyReLU(alpha=0.2)(latent)
        latent=layers.Dropout(0.25)(latent)

        style_modify=layers.Reshape((latent.shape[1],latent.shape[2],embedding_dim))(tf.repeat(encode_style,repeats=latent.shape[1]*latent.shape[2],axis=1))
        latent=layers.concatenate([latent,style_modify],axis=-1)
        latent=layers.Conv2D(64,3,1)(latent)
        # latent=layers.BatchNormalization()(latent)
        latent=layers.LeakyReLU(alpha=0.2)(latent)
        latent=layers.Dropout(0.25)(latent)

        style_modify=layers.Reshape((latent.shape[1],latent.shape[2],embedding_dim))(tf.repeat(encode_style,repeats=latent.shape[1]*latent.shape[2],axis=1))
        latent=layers.concatenate([latent,style_modify],axis=-1)
        latent=layers.Conv2D(128,3,1)(latent)
        # latent=layers.BatchNormalization()(latent)
        latent=layers.LeakyReLU(alpha=0.2)(latent)
        latent=layers.Dropout(0.25)(latent)

        latent=layers.Flatten()(latent)
        style_modify=layers.Reshape((embedding_dim,))(encode_style)
        latent=layers.concatenate([latent,style_modify],axis=-1)

        Judge=layers.Dense(1)(latent)
        # Judge=layers.Activation('sigmoid')(Judge)
        model=Model([musics,style],Judge)

        return model

