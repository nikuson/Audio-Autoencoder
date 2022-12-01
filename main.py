import os
import numpy as np
from scipy.io import wavfile
from keras.layers import Input, Dense
from keras.models import Model

# Set the input and output directories
input_dir = 'input' #this folder can be considered a dataset and at the same time it will process the files from it for output
output_dir = 'output'

# Create a list of all the WAV files in the input directory
wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

# Loop over each WAV file
for wav_file in wav_files:
  # Read the WAV file and convert it to a NumPy array
  rate, data = wavfile.read(os.path.join(input_dir, wav_file))
  data = data.astype('float32') / np.iinfo(data.dtype).max

  # Define the encoder part of the autoencoder
  input_layer = Input(shape=(data.shape[1],))
  encoded = Dense(128, activation='relu')(input_layer)
  encoded = Dense(64, activation='relu')(encoded)

  # Define the decoder part of the autoencoder
  decoded = Dense(64, activation='relu')(encoded)
  decoded = Dense(128, activation='relu')(decoded)
  decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

  # Create the autoencoder model
  autoencoder = Model(input_layer, decoded)

  # Train the autoencoder
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(data, data, epochs=100)

  # Save the encoded and decoded audio to the output directory
  encoded_audio = autoencoder.predict(data)
  wavfile.write(os.path.join(output_dir, wav_file), rate, encoded_audio)
  
  #the code is rather strange, but working. Remember that it is written by a neural network
