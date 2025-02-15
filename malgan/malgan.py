#%% Imports
import os
from dotenv import load_dotenv
import numpy as np
from keras import Input, Model
from keras.api.layers import Dense, Activation, Concatenate
import mlflow

#%% Constants
load_dotenv()
NUM_FEATURES=69
BATCH_SIZE=64
NOISE_DIM=NUM_FEATURES
DEBUG=int(os.getenv("DEBUG"))

#%%
def Generator(input_shape=NUM_FEATURES, noise_shape=NOISE_DIM):
    input = Input(shape=[input_shape])
    noise = Input(shape=[noise_shape])
    x = Concatenate(axis=1)([input, noise])

    x = Dense(units=x.shape[1])(x)
    x = Activation(activation='relu')(x)

    x = Dense(units=128)(x)
    x = Activation(activation='relu')(x)

    x = Dense(units=input.shape[1])(x)
    output = Activation(activation='sigmoid')(x)

    model = Model([input, noise], output)
    return model

if DEBUG == 1:
    generator = Generator()
    print(generator.summary())
#%% Building the substitute detector model from scratch
def SubstituteDetector(input_shape=NUM_FEATURES):
    input = Input(shape=[input_shape])

    x = Dense(units=input.shape[1])(input)
    x = Activation(activation='sigmoid')(x)

    x = Dense(units=256)(x)
    x = Activation(activation='sigmoid')(x)

    x = Dense(units=32)(x)
    x = Activation(activation='sigmoid')(x)

    x = Dense(units=1)(x)
    output = Activation(activation='sigmoid')(x)

    model = Model(input, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if DEBUG == 1:
    substituteDetector = SubstituteDetector()
    print(substituteDetector.summary())

#%% MalGAN model
def MalGAN(generator, substituteDetector, input_shape=NUM_FEATURES, noise_shape=NOISE_DIM):
    for layer in substituteDetector.layers:
        layer.trainable = False

    input = Input(shape=[input_shape])
    noise = Input(shape=[noise_shape])

    x = generator([input, noise])
    output = substituteDetector(x)

    model = Model([input, noise], [output, x])
    model.compile(optimizer='adam', loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100])
    return model

if DEBUG == 1:
    malGAN = MalGAN(generator, substituteDetector)
    print(malGAN.summary())
    
# %% Helper functions
def getMalGAN(num_features):
    generator, substDetector = Generator(num_features, num_features), SubstituteDetector(num_features)
    return generator, substDetector, MalGAN(generator, substDetector)

def getLabeledData(generator, blackBox, malware_batch, benign_batch, noise_batch):
    X = generator.predict([malware_batch, noise_batch])
    samples = np.concatenate([benign_batch, X], axis = 0)
    np.random.shuffle(samples)
    bb_predictions = blackBox.predict(samples)
    return samples, bb_predictions

def get_noise(batch_size, noise_dim):
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    return noise

def getTestData(generator, malware, benign, noise_dim=NOISE_DIM):
    noise = get_noise(malware.shape[0], noise_dim)
    synthetic = generator.predict([malware, noise])

    test_data = np.concatenate((np.concatenate((benign, malware), axis=0), synthetic), axis=0)

    test_labels = list()
    test_labels.extend([0] * benign.shape[0])
    test_labels.extend([1] * malware.shape[0])
    test_labels.extend([1] * synthetic.shape[0] * synthetic.shape[1])
    test_labels = np.array(test_labels)

    shuffle = np.arange(test_data.shape[0])
    np.random.shuffle(shuffle)
    test_data = test_data[shuffle]
    test_labels = test_labels[shuffle]

    return test_data, test_labels

#%% Training MalGAN
def train(generator, blackBox, substituteDetector, malGAN, malware, benign, epochs=5, noise_dim=NOISE_DIM):
    mlflow.keras.autolog()

    benign_batch_idx = 0
    malware_batch_count = malware.shape[0]
    benign_batch_count = benign.shape[0]

    for epoch in range(epochs):
        sd_losses = []
        gen_losses = []
        for malware_batch_idx in range(malware_batch_count):
            malware_batch = malware[malware_batch_idx]
            benign_batch = benign[benign_batch_idx]
            benign_batch_idx = (benign_batch_idx + 1) % benign_batch_count
            noise_batch = get_noise(malware_batch.shape[0], noise_dim)

            bb_samples, bb_labels = getLabeledData(generator, blackBox, malware_batch, benign_batch, noise_batch)
            subsitituteDetector_loss = substituteDetector.train_on_batch(bb_samples, bb_labels)
            sd_losses.append(subsitituteDetector_loss)
            
            sd_out = substituteDetector.predict(malware_batch)
            generator_loss = malGAN.train_on_batch([malware_batch, noise_batch], [sd_out, malware_batch])
            gen_losses.append(generator_loss)
            
        mlflow.log_metric("avg_subst_loss", np.average(sd_losses), step=epoch)
        mlflow.log_metric("avg_gen_loss", np.average(gen_losses), step=epoch)
        print(epoch, subsitituteDetector_loss, generator_loss)
        