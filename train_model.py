import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
TRAINING_DIR = "data/WR/train/image"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                              batch_size=40,
                              class_mode='binary',
                              target_size=(278,278))

VALIDATION_DIR = "data/WR/test/image"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                      batch_size=40,
                                      class_mode='binary',
                                      target_size=(278,278))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                           input_shape=(278,278, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(35, activation='softmax')
])
model.compile(optimizer='adam', 
loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator,
                   epochs=20,
                   validation_data=validation_generator)

model.save('model.h5')

import matplotlib.pyplot as plt

# Получаем список потерь и точности из истории обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Создаем фигуру и оси для графика
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig("test2.png")
plt.show()


