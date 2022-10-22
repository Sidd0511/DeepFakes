from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from glob import glob
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_name = 'VGG16'

IMAGE_SIZE = (224, 224, 3)

""" GENERATING THE DATASET  """

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory('Training Set/',
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('Test Set/',
                                            target_size=(224, 224),
                                            batch_size=64,
                                            class_mode='categorical',
                                            shuffle=True)

validation_set = test_datagen.flow_from_directory('Validation Set/',
                                                  target_size=(224, 224),
                                                  batch_size=1,
                                                  class_mode='categorical',
                                                  shuffle=False)

"""  LOADING THE PRE-TRAINED MODEL  """
base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=IMAGE_SIZE)

total_layers = len(base_model.layers)
untrain_layers = total_layers - int(total_layers * 0.30)

for layer in base_model.layers[:untrain_layers]:
    layer.trainable = False

for layer in base_model.layers[untrain_layers:]:
    layer.trainable = True

"""  ADD AVERAGE POOLING LAYER AND FULLY CONNECTED LAYER """
top_layer = GlobalAveragePooling2D()(base_model.output)

top_layer = Dense(units=1024, activation='relu')(top_layer)
top_layer = Dropout(rate=0.2)(top_layer)

top_layer = Dense(units=1024, activation='relu', name='fc2')(top_layer)
top_layer = Dropout(rate=0.2)(top_layer)

top_layer = Dense(units=256, activation='relu', name='fc3')(top_layer)
top_layer = Dropout(rate=0.2)(top_layer)

output_layer = Dense(units=2, activation='softmax', name='output')(top_layer)

model = Model(inputs=base_model.input, outputs=output_layer)

# model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

""" CALLBACKS """
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

checkpoint = ModelCheckpoint(filepath=model_name+'.weights.best.hdf5',
                             save_best_only=True,
                             verbose=1)

r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    verbose=2,
    callbacks=[early_stop, checkpoint]
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig(model_name+'_LossVal_loss')

# accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig(model_name+'_AccVal_acc')


"""  MAKING PREDICTIONS  """
print('\n\n\n')
model.load_weights(model_name+'weights.best.hdf5')  # initialize the best trained weights

true_classes = validation_set.classes
class_indices = training_set.class_indices
class_indices = dict((v, k) for k, v in class_indices.items())

predictions = model.predict(validation_set)
prediction_classes = np.argmax(predictions, axis=1)

model_acc = accuracy_score(true_classes, prediction_classes)
print(f"{model_name} Model Accuracy : {:.2f}%".format(model_acc * 100))


"""  CONFUSION MATRIX   """
class_names = validation_set.class_indices.keys()

def plot_heatmap(y_true, y_pred, class_names, ax, title, fmt):
    cm = confusion_matrix(y_true, y_pred,)
    sns.heatmap(
        cm,
        annot=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt=fmt,
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

plot_heatmap(true_classes, prediction_classes, class_names, ax1, title="Transfer Learning (VGG16) No Fine-Tuning",fmt='.2%')
plot_heatmap(true_classes, prediction_classes, class_names, ax2, title="Transfer Learning (VGG16) No Fine-Tuning",fmt='d')

fig.suptitle(f'{model_name} + Confusion Matrix')
fig.tight_layout()
fig.subplots_adjust(top=1.25)
plt.show()
plt.savefig(f'{model_name} + Confusion Matrix')