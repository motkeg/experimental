import autokeras as ak
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

EPOCHS = 50
BATCH = 5
NAME = "autokeras_classification"
DATASET_PATH = "/hdd/4celebs_training_set"


def build_train_set(image_size):
    def make_train_generator():
        data_generator = image.ImageDataGenerator(rescale=1. / 255,
                                                  # shear_range=0.2,
                                                  zoom_range=0.1,
                                                  horizontal_flip=True,
                                                  validation_split=0.15,
                                                  width_shift_range=0.2,
                                                  height_shift_range=0.2)

        train_generator = data_generator.flow_from_directory(
            DATASET_PATH,
            target_size=(image_size, image_size),
            batch_size=BATCH,
            class_mode='categorical',
            subset='training')
        return train_generator

    return tf.data.Dataset.from_generator(make_train_generator, (tf.float16, tf.float16))


def build_val_set(image_size):
    def make_val_generator():
        data_val_gen = image.ImageDataGenerator(rescale=1. / 255,
                                                validation_split=0.1)

        validation_generator = data_val_gen.flow_from_directory(
            DATASET_PATH,
            target_size=(image_size, image_size),
            class_mode='categorical',
            subset='validation')
        return validation_generator

    return tf.data.Dataset.from_generator(make_val_generator, (tf.float32, tf.float32))


def build_model():
    model = ak.ImageClassifier(max_trials=100, objective="val_acc")
    return model


def train(model, image_size):
    tensorboard = TensorBoard(log_dir=f'output/logs/{NAME}',
                              write_graph=True,
                              # histogram_freq=5,
                              write_images=True,
                              write_grads=True,
                              profile_batch=3)

    checkpointer = ModelCheckpoint(
        filepath=f'output/weights/{NAME}_clooney.best.hdf5',
        verbose=1,
        save_best_only=True,
        monitor='val_acc')

    train_set = build_train_set(image_size)
    # val_set = build_val_set(image_size)
    model.fit(
        train_set,
        steps_per_epoch=EPOCHS * BATCH,
        epochs=EPOCHS,
        # validation_data=val_set,
        validation_split=0.15,
        verbose=2,
        callbacks=[checkpointer, tensorboard])


if __name__ == '__main__':
    model = build_model()
    train(model, image_size=312)
