
from pathlib import Path
from keras import layers
import joblib
import time
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def export_model(mode=None):
    BATCH_SIZE = 8
    TARGET_SIZE = 224  # Based on EfficientNetB0
    NUM_CLASSES = 10
    img_data = str(Path(__file__).parent.parent)+"\data\images_original"
    dir = str(Path(__file__).parent.parent)
    train_ds = image_dataset_from_directory(
        img_data,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE)
    val_ds = image_dataset_from_directory(
        img_data,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE)
    class_names = train_ds.class_names
    model_save = tf.keras.callbacks.ModelCheckpoint('./best_weights.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='min', verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                  patience=10, mode='min', verbose=1,
                                                  restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                     patience=2, min_delta=0.001,
                                                     mode='min', verbose=1)

    def create_model():
        conv_base = EfficientNetB0(include_top=False, weights="imagenet", drop_connect_rate=0.6,
                                   input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
        model = conv_base.output
        model = layers.GlobalAveragePooling2D()(model)
        model = layers.Dense(NUM_CLASSES, activation="softmax")(model)
        model = models.Model(conv_base.input, model)

        model.compile(optimizer=Adam(lr=0.001),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    model = create_model()

    epochs = 25
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_save, early_stop, reduce_lr],
        verbose=2
    )

    # 모델 저장(첫 학습, 재 학습 구분)
    if not mode:
        model.save(dir+"/model/model.h5")
    else:
        if os.path.isfile(dir + '/model/model.h5'):
            os.rename(dir + '/model/model.h5', dir + f'/model/model_{time.time()}.h5')
        model.save(dir+"/model/model.h5")


if __name__ == "__main__":
    export_model()