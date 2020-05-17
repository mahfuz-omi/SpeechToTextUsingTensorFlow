import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from playsound import playsound
import warnings
warnings.filterwarnings("ignore")

# loading saved model
def load_saved_model():
    from keras.models import load_model
    model = load_model(r"best_model.hdf5")
    return model

def create_model():
    # creating keras model
    from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
    from keras.models import Model
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras import backend as K
    K.clear_session()

    inputs = Input(shape=(8000, 1))

    # First Conv1D layer
    conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Flatten layer
    conv = Flatten()(conv)

    # Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    # Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(classes), activation='softmax')(conv)

    model = Model(inputs, outputs)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and to save the best model after every epoch:

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    # Epoch 00001: val_acc improved from -inf to 0.60268, saving model to best_model.hdf5
    # fit function execute hobar por model ta best_model.hdf5 name e save hobe
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # Epoch 00051: val_acc did not improve from 0.95257
    # Epoch 00051: early stopping
    # now predict new live data
    # ager trained model ta saved hoye geche
    model.fit(X_train, y_train, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(X_test, y_test))
    return model

#Define the function that predicts text for the given audio:
def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1), verbose=0).round(decimals=2)
    index=np.argmax(prob[0])
    return classes[index]

def random_audio_test():
    import librosa
    from playsound import playsound

    #os.chdir(r"F:\python codes\interview_codes\speech_recognition_data\train\zero")
    random_file = "F:\python codes\interview_codes\zero.wav"
    playsound(random_file)

    y, sr = librosa.load(random_file, sr=8000)

    print(len(y))

    print("Text:", predict(y))


# returns like mnist dataset from keras
def load_data():
    X_train = []
    y_train = []

    X_test = []
    y_test = []


    # training
    # train cat
    i = 0
    fileNameList = os.listdir(r"speech_recognition_data\train\cat")
    os.chdir(r"speech_recognition_data\train\cat")

    for fileName in fileNameList:
        speech_array, sr = librosa.load(fileName,sr=8000)

        if(len(speech_array) == 8000):
            if(i==4):
                X_test.append(speech_array.tolist())
                y_test.append('cat')
                i = 0
            else:
                X_train.append(speech_array.tolist())
                y_train.append('cat')
                i = i + 1

    # training
    # train dog
    i = 0
    fileNameList = os.listdir(r"speech_recognition_data\train\dog")
    os.chdir(r"speech_recognition_data\train\dog")

    for fileName in fileNameList:
        speech_array, sr = librosa.load(fileName, sr=8000)

        if (len(speech_array) == 8000):
            if (i == 4):
                X_test.append(speech_array.tolist())
                y_test.append('dog')
                i = 0
            else:
                X_train.append(speech_array.tolist())
                y_train.append('dog')
                i = i + 1


    # training
    # train happy
    i = 0
    fileNameList = os.listdir(r"speech_recognition_data\train\happy")
    os.chdir(r"speech_recognition_data\train\happy")

    for fileName in fileNameList:
        speech_array, sr = librosa.load(fileName, sr=8000)

        if (len(speech_array) == 8000):
            if (i == 4):
                X_test.append(speech_array.tolist())
                y_test.append('happy')
                i = 0
            else:
                X_train.append(speech_array.tolist())
                y_train.append('happy')
                i = i + 1


    # training
    # train right
    i = 0
    fileNameList = os.listdir(r"speech_recognition_data\train\right")
    os.chdir(r"speech_recognition_data\train\right")

    for fileName in fileNameList:
        speech_array, sr = librosa.load(fileName, sr=8000)

        if (len(speech_array) == 8000):
            if (i == 4):
                X_test.append(speech_array.tolist())
                y_test.append('right')
                i = 0
            else:
                X_train.append(speech_array.tolist())
                y_train.append('right')
                i = i + 1


    # training
    # train zero
    i = 0
    fileNameList = os.listdir(r"speech_recognition_data\train\zero")
    os.chdir(r"speech_recognition_data\train\zero")

    for fileName in fileNameList:
        speech_array, sr = librosa.load(fileName, sr=8000)

        if (len(speech_array) == 8000):
            if (i == 4):
                X_test.append(speech_array.tolist())
                y_test.append('zero')
                i = 0
            else:
                X_train.append(speech_array.tolist())
                y_train.append('zero')
                i = i + 1

    return (X_train, y_train), (X_test, y_test)



(X_train, y_train),(X_test, y_test) = load_data()

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape,X_test.shape)
# (7177, 8000) (1792, 8000)

# i-hot encoding the y data
# before, convert the categorial data to numeric form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
classes= list(le.classes_)

# print('before 1-hot encoding:')
# print(y_train)
# print(y_test)

from keras.utils import to_categorical
num_classes = len(classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# print("after to_categorical: ",y_train)
# print("after to_categorical: ",y_test)


# reshaping X
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model = load_saved_model()
random_audio_test()








