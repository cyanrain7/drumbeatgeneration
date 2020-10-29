import datetime
import numpy as np
import copy
import tensorflow as tf
import sys


# new added functions for cyclegan
class ImagePool(object):

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.  # 64 * 84 * 1
    npy_B = np.load(npy_data[1]) * 1.  # 64 * 84 * 1
    npy_AB = np.concatenate((npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
                             npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1)),
                            axis=2)  # 64 * 84 * 2
    return npy_AB


def save_midis(bars, file_path, tempo=80.0):
    sys.path.append('../google_project')
    import data
    from midi_io import midi_file_to_note_sequence,note_sequence_to_midi_file

    # print(file_path)
    # exit()

    music_length=32
    data_converter=data.GrooveConverter(
        split_bars=music_length//16,
        steps_per_quarter=4,
        quarters_per_bar= 4,
        max_tensors_per_notesequence=50, 
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES, 
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES)
    
    outputs = []
    for i, bar in enumerate(bars):
        output_hits = (np.array(bar)).reshape(32,12)
        output_hits = output_hits[:32, :9]
        output_velocities = np.ones(output_hits.shape)*output_hits
        output_offsets = np.zeros(output_hits.shape)
        output=tf.concat([output_hits,output_velocities,output_offsets],axis=-1)
        outputs.append(output)
    
    seq = data_converter._to_notesequences(outputs)
    for i,s in enumerate(seq):
        sn = file_path.replace('.mid', f'_{i}.mid')
        note_sequence_to_midi_file(s, sn)


def get_now_datetime():
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return str(now)


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keepdims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track
