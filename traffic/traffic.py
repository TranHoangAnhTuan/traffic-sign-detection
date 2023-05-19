import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    tempImage, labels = load_data(sys.argv[1])
    # labels = tf.keras.utils.to_categorical(labels)
    tempImage = np.array(tempImage)

    images = tempImage / 255
    print(labels)
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # y_train = tf.keras.utils.to_categorical(y_train, NUM_CATEGORIES)
    # y_test = tf.keras.utils.to_categorical(y_test, NUM_CATEGORIES)
    # Get a compiled neural network
    model = get_model()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    image = []
    img_path = "C:/working/cs50AI/week5/test_images/two_cars.ppm"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (30,30))
    image.append(img)
    image = np.array(image)
    image = image / 255
    prediction = model.predict(image)


# Print the predicted class
    predicted_class = np.argmax(prediction)
    print(f"Predicted class: {predicted_class}")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    import resize_image

    images = []
    labels = []
    res = []

    # Loop over directory indexes from 0 to 43
    for i in range(NUM_CATEGORIES):
        # Set directory path for current index
        dir_path = os.path.join(data_dir, str(i))
        print(dir_path)
        
        # Get list of image file names in directory
        img_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.ppm')]

        # Loop through image files in directory and read and resize each image
        for img_file in img_files:
            img_path = os.path.join(dir_path, img_file)
            img = resize_image.resize(img_path)
            img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH))
            images.append(img)
            labels.append(i)

    # Convert images and labels lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer with 32 filters and a 3x3 kernel size
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Max pooling layer with a 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten layer to convert output from 2D to 1D
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation='relu'),
        # Dense layer with NUM_CATEGORIES output units and softmax activation for classification
        tf.keras.layers.Dense(units=NUM_CATEGORIES, activation='softmax'),
        tf.keras.layers.Dropout(0.5)
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    return model



if __name__ == "__main__":
    main()
