import os
import wandb
import tensorflow as tf
from datetime import datetime


wandb.tensorboard.patch(root_logdir="./logs/debug")
wandb.init(project="tensorboard-demo", sync_tensorboard=True)

tf.debugging.experimental.enable_dump_debug_info(
    "./logs/debug", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
)

labels = [
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
(
    (train_images, train_labels),
    (test_images, test_labels),
) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
)

model.fit(
    train_images,
    train_labels,
    batch_size=64,
    validation_data=(test_images, test_labels),
    validation_batch_size=64,
    epochs=1,
    callbacks=[tensorboard_callback],
)
