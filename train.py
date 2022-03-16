import os

import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from datetime import datetime


wandb.login()
wandb.init(project="tensorboard-demo", sync_tensorboard=True)

config = wandb.config
config.labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
config.input_size = 28
config.dense_layer_units = 32
config.dropout_rate = 0.2
config.batch_size = 64
config.validation_batch_size = 64
config.learning_rate = 1e-3
config.epochs = 5
config.tensorboard_log_dir = os.path.join(
    "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
)

(
    (train_images, train_labels),
    (test_images, test_labels,),
) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(config.input_size, config.input_size)),
        tf.keras.layers.Dense(config.dense_layer_units, activation="relu"),
        tf.keras.layers.Dropout(config.dropout_rate),
        tf.keras.layers.Dense(len(config.labels), activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=config.tensorboard_log_dir
)

wandb_callback = WandbCallback(
    log_evaluation=True, validation_steps=config.validation_batch_size
)

model.fit(
    train_images,
    train_labels,
    batch_size=config.batch_size,
    validation_data=(test_images, test_labels),
    validation_batch_size=config.validation_batch_size,
    epochs=config.epochs,
    callbacks=[tensorboard_callback, wandb_callback],
)

wandb.finish()
