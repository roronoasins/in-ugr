import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.utils import np_utils

emnist = ["emnist/letters", "emnist/balanced", "emnist/bymerge", "emnist/byclass"]
dataset =  emnist[2]
input_shape = (28,28, 1)
batch_size = 64
epochs = 10
if dataset == emnist[0]:
    num_classes = 26
    epochs = 10
else:
    num_classes = 47



#train = tfds.load(name="emnist/letters", split="train[:80%]", as_supervised=True)
train = tfds.load(name=dataset, split="train", as_supervised=True)
#val = tfds.load(name="emnist/letters", split="train[:-20%]")
test = tfds.load(name=dataset, split="test", as_supervised=True)

train_numpy = np.vstack(tfds.as_numpy(train))
#val_numpy = np.vstack(tfds.as_numpy(val))
test_numpy = np.vstack(tfds.as_numpy(test))

train_x = np.array(list(map(lambda x: x[0], train_numpy)))
train_y = np.array(list(map(lambda x: x[1], train_numpy)))
#val_x = np.array(list(map(lambda x: x[0], val_numpy)))
#val_y = np.array(list(map(lambda x: x[1], val_numpy)))
test_x = np.array(list(map(lambda x: x[0], test_numpy)))
test_y = np.array(list(map(lambda x: x[1], test_numpy)))

if dataset == emnist[0]:
    train_y -= 1
    test_y -= 1

train_x = train_x.astype('float32') / 255.
#val_x = val_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.

l = tf.keras.layers
model = tf.keras.Sequential([
    l.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',input_shape=input_shape),
    #l.AveragePooling2D(),
    l.MaxPooling2D(2,2),
    l.Dropout(0.4),
    l.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',input_shape=input_shape),
    l.MaxPooling2D(2,2),
    l.Dropout(0.4),
    l.Flatten(input_shape=input_shape),
    l.Dense(units=512, activation='relu'),
    l.Dropout(0.4),
    l.Dense(num_classes,activation='softmax')
])
model.summary()

optimizer = ['adam', 'rmsprop']
model.compile(optimizer=optimizer[0],loss='sparse_categorical_crossentropy',metrics=['accuracy'])

MCP = ModelCheckpoint('best_points_balanced.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights=True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

history = model.fit(
        train_x,train_y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        #validation_data=(val_x,val_y),
        validation_split=0.2,
        callbacks=[MCP,ES,RLP]
        )

test_loss, test_score = model.evaluate(test_x,test_y, verbose=0)

print('Test loss:', test_loss)
print('Test accuracy:', test_score)
