import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# Create Convolutional neural network

model = models.Sequential()
# Layer 1
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 2
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 3
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation='relu', input_shape=(32, 32, 3)))

# Fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

model.summary()


#Compile and train NN
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=15,
                    validation_data=(test_images, test_labels))


# Evaluate NN
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# Individual Image Test
# select the image from our test dataset
image_number = 40

# display the image
plt.imshow(test_images[image_number])

# load the image in an array and reshape it
import numpy as np
n = np.array(test_images[image_number])
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and save the predicted label
predicted_label = class_names[model.predict(p).argmax()]

# load the original label
original_label = class_names[test_labels[image_number][0]]

# display the result
print("Original label is {} and predicted label is {}".format(
    original_label, predicted_label))