import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from scipy.io import wavfile
import sklearn
from sklearn.metrics import mean_squared_error

#imports data but only of given sample rate
def import_data(DATADIR, sample_rate):
  
  #import data and sample rate

  training_data=[]

  for file in os.listdir(DATADIR): 
    path= os.path.join(DATADIR,file)
    if os.path.isdir(path):  
      for sound in os.listdir(path):
        # Check whether file is in wav format or not 
        if sound.endswith(".wav"): 
          try:
            sound_path = f"{path}/{sound}"
            samplerate, data = wavfile.read(sound_path)
            training_data.append([samplerate,data])
          except:
            pass # doing nothing on exception
    else:
      if file.endswith(".wav"):
        try:
          samplerate, data = wavfile.read(path)
          training_data.append([samplerate,data])
        except:
          pass

  #only keep files of given sample rate and remove sample rate column
  a=0
  x=[]
  for i in range(len(training_data)):
    if training_data[i][0] == sample_rate:
      x.append(training_data[i][1])
    else:
      a += 1
  print("Kept files with " + str(sample_rate) + " and deleted "+ str(a) + " files with different sample rate")
  return x


#returns a list with all the sample rates for a tuple of sample rates, data
def return_sample_rates(data):
  x=[]
  for i in range(len(data)):
    x.append(data[i][0])
  return x


def plot_lengths(training_data):
  
  ## create a list of number of samples per file in order to find optimal length to bring them all to
  lengths=[]
  for i in range(0,len(training_data)):
    lengths.append(len(training_data[i]))

  #do histogram of length of samples in order to do this (see above)
  plt.hist(lengths)
  plt.show()
  
  print("The maximum length of samples for this dataset is "+ str(max(lengths)) + " and the average samples per wav file is " +str(np.mean(lengths)))
  

def reshape_data(data,max_length,BATCH_SIZE,BUFFER_SIZE):
  #reshape in right shape for input to neural network

  train_dataset = tf.data.Dataset.from_tensor_slices(data)
  train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  
  return train_dataset


def write_wav_data(data,destination,sample_rate=44100):
  #check if destination folder exists, otherwise create
  newpath = destination
  if not os.path.exists(newpath):
    os.makedirs(newpath)

  #write into new dataset
  for i in range(len(data)):
    scipy.io.wavfile.write(str(destination)+str(i)+".wav", sample_rate, data[i])
    
    
def find_most_similar_audio(audio,dataset):
  #uses mean square error to compare the generated data to the 
  x=[]
  for i in range(0,len(dataset)):
    x.append(mean_squared_error(audio, dataset[i]))
  min_value=min(x)
  min_index= np.argmin(x)
  print("The most similar audio in the dataset is number {} (index nr. {}) and has an mse of {}".format(str(min_index+1),str(min_index),str(min_value)))
  return min_index







def stereo_to_mono(data):
  #takes both channels and returns a single one that is the average of both
  training_data_mono=[]
  x=data
  a=0
  for i in range(len(x)):
    if x[i][0].shape == (2,):
      z=[]
      for j in range(len(x[i])):
        z.append(np.mean(x[i][j]))
      z=np.array(z)
      training_data_mono.append(z)
      a += 1
    else:
      training_data_mono.append(x[i])
  return training_data_mono
  print("Number of stereo files have been successfully converted into mono= "+ str(a)+"\n")

def normalize_audio(data):
  #normalizes data to values between -1 and +1
  for i in range(len(data)):
    m,f= max(data[i]), min(data[i])
    f=f*-1
    y=max(m,f)
    data[i]=data[i]/y
  print("Normalization to values between -1 and +1 completed successfully \n")
  return data

def make_into_list(data):
  training_data_list=[]
  training_data_mono=data
  for i in range(len(training_data_mono)):
    x=[]
    for j in range(len(training_data_mono[i])):
      x.append(training_data_mono[i][j])
    training_data_list.append(x)
  return training_data_list

def strip_silence(training_data_mon):
  ## This function removes all zeros except for one from the beginning and from the end of the file, 
  ## reducing its size in the case of files with a lot of silence on both ends
  ## This also helps to align the different training snare sounds 

  ## Strip silence at beginning of files 
  for i in range(len(training_data_mon)):
    for j in range(len(training_data_mon[i])):
      if abs(training_data_mon[i][0])<=0.001 and abs(training_data_mon[i][1])<=0.001:
        del(training_data_mon[i][0])
      else: 
        break
  ## Strip silence at end of files
  for i in range(len(training_data_mon)):
    for j in range(len(training_data_mon[i])):
      if abs(training_data_mon[i][len(training_data_mon[i])-1])<=0.001 and abs(training_data_mon[i][len(training_data_mon[i])-2])<=0.001 :
        del(training_data_mon[i][len(training_data_mon[i])-1])
      else: 
        break
  return training_data_mon

def remove_long_files(data,max_length):
  x=[]
  for i in range(len(data)):
    if len(data[i])<=max_length:
      x.append(data[i])
  print("Files with length of over "+str(max_length)+ " have been successfully removed \n")
  data_without_long=x
  return data_without_long

def pad_data(data,desired_length):
  for i in range(len(data)):
    while len(data[i])<desired_length:
      data[i].append(0)
  print("Padding to achieve desired length \n")
  return data

def list_into_float32arr(data):
  for i in range(len(data)):
    data[i]=np.array(data[i])
  #turn into floatpoint32
  fl32_data=[]
  for i in range(len(data)):
    fl32_data.append(data[i].astype(np.float32))
  return fl32_data

def preprocess_data(training_data,max_length):
  
  #go from stereo to mono
  training_data_mono=stereo_to_mono(training_data)
  
  #convert to float32
  training_data_mono= list_into_float32arr(training_data_mono)
  #normalize data so that it is between -1 and +1 
  normal_mono_data=normalize_audio(training_data_mono)
  
  #calculate lengths before removing silence
  lengths=[]
  for i in range(0,len(normal_mono_data)):
    lengths.append(len(normal_mono_data[i]))
  
  #turn back into python list otherwise next part of the code doesn't work
  data_list=make_into_list(normal_mono_data)

  ## Strip silence
  stripped_data=strip_silence(data_list)

  ## calculate lengths after having stripped silences
  lengthsafter=[]
  for i in range(0,len(stripped_data)):
    lengthsafter.append(len(stripped_data[i]))

  print("Successfully stripped all beginning and end silences effectively reducing the average samples of the dataset from "+ str(np.mean(lengths))+ " to " + str(np.mean(lengthsafter))+"\n")
  
  #remove long files 
  no_long_data= remove_long_files(stripped_data,max_length)
  
  ##PAD WITH ZEROS UNTIL ALL SAME LENGTH
  padded_data=pad_data(no_long_data,max_length)

  #turn list elements into numpy floatpoint32 array
  final_data=list_into_float32arr(padded_data)

  #turn into array
  data_np = np.asarray(final_data, np.float32)
  data_np= data_np.reshape(len(final_data),max_length,1)

  return data_np

