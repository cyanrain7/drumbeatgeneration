import csv
import numpy as np
from midi_io import midi_file_to_note_sequence,note_sequence_to_midi_file
import data
import os
style=['rock','latin','soul','reggae','neworleans','blues','afrocuban','funk','jazz','hiphop','country','afrobeat','dance','punk','gospel','middleeastern','pop','highlife']
label_map={'rock':1,'latin':2,'funk':3,'jazz':4}
recorder={}
seq_length=32
for key in label_map.keys():
    recorder[key]=0
data_converter=data.GrooveConverter(
        split_bars=seq_length//16,
        steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=50,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES)
interval=4
def datapreprocess():
    midi_list=csv.reader(open(".\groove\info.csv",'r'))
    dataset=[]
    
    for key,value in label_map.items():
        print(key,value)
        file_list=os.listdir('./dataset_check/'+key)
        count=0
        for file in file_list:
            
            sequence=midi_file_to_note_sequence('./dataset_check/'+key+'/'+file)
            tensor=data_converter.to_tensors(note_sequence=sequence)
            collector=[]
            for i,tensor_slice in enumerate(tensor.outputs):
                collector.append(tensor_slice)
                if i==0:
                    last_right=tensor_slice[seq_length//2:]
                else:
                    new_slice=np.concatenate([last_right,tensor_slice[:seq_length//2]],axis=0)
                    last_right=tensor_slice[seq_length//2:]
                    collector.append(new_slice)
            for tensor_slice in collector:
                flag=False
                for p in range(0,seq_length-interval+1):
                    if np.sum(tensor_slice[p:p+interval,:9])==0:
                        flag=True
                        break
                if flag:
                    continue
                dataset.append(np.concatenate((tensor_slice.flatten(),[value])))
                count+=1
        recorder[key]=count
    print(len(dataset))
    np.save('Groove',np.array(dataset))

if __name__ == "__main__":
    datapreprocess()