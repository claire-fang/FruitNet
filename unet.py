import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from PIL import Image, ImageOps
from io import BytesIO

import matplotlib.pyplot as plt

def convert_yolo_box_to_corners(x_center, y_center, box_w, box_h, img_width, img_height):
    # Ensure everything is float for math
    x_center = tf.cast(x_center, tf.float32)
    y_center = tf.cast(y_center, tf.float32)
    box_w    = tf.cast(box_w, tf.float32)
    box_h    = tf.cast(box_h, tf.float32)

    img_width  = tf.cast(img_width, tf.float32)
    img_height = tf.cast(img_height, tf.float32)

    # Convert normalized YOLO coords → pixels
    x_center = x_center * img_width
    y_center = y_center * img_height
    box_w    = box_w * img_width
    box_h    = box_h * img_height

    # Compute pixel corners
    x1 = tf.cast(x_center - box_w / 2, tf.int32)
    y1 = tf.cast(y_center - box_h / 2, tf.int32)
    x2 = tf.cast(x_center + box_w / 2, tf.int32)
    y2 = tf.cast(y_center + box_h / 2, tf.int32)

    # Safety clipping
    x1 = tf.clip_by_value(x1, 0, tf.cast(img_width,  tf.int32)  - 1)
    y1 = tf.clip_by_value(y1, 0, tf.cast(img_height, tf.int32) - 1)
    x2 = tf.clip_by_value(x2, 0, tf.cast(img_width,  tf.int32)  - 1)
    y2 = tf.clip_by_value(y2, 0, tf.cast(img_height, tf.int32) - 1)

    # Return x,y,w,h for your crop slicing
    return x1, y1, (x2-x1), (y2-y1)

import pandas as pd
import cv2
import os

# unet_crop paths
# img_path = f"{root_path}/unet_crops/images/{row["mask_path"]}"
# mask_path = f"{root_path}/unet_crops/masks/{row["mask_path"]}"

# whole_food_paths
# img_path = f"{root_path}/images/{row["image_path"]}"
# mask_path = f"{root_path}/masks/{image_name}/{row["mask_path"]}"

# data124
# img_path = f"{root_path}/images/{row["image_path"]}"
# mask_path = f"{root_path}/masks/{image_name}/{row["mask_path"]}"

# data124 test
# img_path = f"{root_path}/images/{row["image_path"]}"
# mask_path = f"{root_path}/masks_agg/{row["file_name"]}.png"


def load_and_crop_training_data(root_path):
    # Read CSV
    csv_path = os.path.join(root_path, "source_annotations.csv")
    df = pd.read_csv(csv_path)

    # Filter training rows
    filter_df = df[(df["warm_color_binary"] == 1) & (df["mask_path"].notna())]
    train_df = filter_df[df["train_test_validation"] == 0]
    dev_df = filter_df[df["train_test_validation"] == 1]
    test_df = filter_df[df["train_test_validation"] == 2]

    def get_img_mask_bounding_info(df):
      image_list = []
      mask_list = []
      bounding_list = []

      for _, row in df.iterrows():
          image_name = row["file_name"]
          img_path = f"{root_path}/images/{row["image_path"]}"
          mask_path = f"{root_path}/masks_agg/{row["file_name"]}.png"

          image_list.append(img_path)
          mask_list.append(mask_path)

          bounding_list.append((row["x_center"], row["y_center"], row["width"], row["height"]))

      return image_list, mask_list, bounding_list

    train_image_list, train_mask_list, train_bounding_list = get_img_mask_bounding_info(train_df)
    dev_image_list, dev_mask_list, dev_bounding_list = get_img_mask_bounding_info(dev_df)
    test_image_list, test_mask_list, test_bounding_list = get_img_mask_bounding_info(test_df)

    return train_image_list, train_mask_list, train_bounding_list, dev_image_list, dev_mask_list, dev_bounding_list, test_image_list, test_mask_list, test_bounding_list


# Image
def build_dataset(image_list, mask_list, bounding_list):
  image_filenames = tf.constant(image_list)
  masks_filenames = tf.constant(mask_list)
  boounding_boxes = tf.constant(bounding_list)

  dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames, boounding_boxes))
  return dataset

def load_image_with_orientation(image_path):
  img_bytes = tf.io.read_file(image_path)
  img = Image.open(BytesIO(img_bytes.numpy()))
  img = ImageOps.exif_transpose(img)
  img = img.convert('RGB')
  width, height = img.size
  img = np.array(img).astype(np.float32) / 255.0
  return img, np.int32(width), np.int32(height)

def process_path(image_path, mask_path, bounding_box):
    img, width, height = tf.py_function(load_image_with_orientation, [image_path], [tf.float32, tf.int32, tf.int32])
    img.set_shape([None, None, 3])
    img = tf.reshape(img, (height, width, 3))
    crop_window = convert_yolo_box_to_corners(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], width, height)
    # img = tf.io.read_file(image_path)
    # img_shape = tf.image.extract_jpeg_shape(img)
    # crop_window = convert_yolo_box_to_corners(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], img_shape[1], img_shape[0])
    # img = tf.image.decode_jpeg(img, channels=3)
    img = img[crop_window[1]:crop_window[1]+crop_window[3] + 1, crop_window[0]:crop_window[0]+crop_window[2] + 1, :]
    # img = tf.image.decode_png(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = mask[crop_window[1]:crop_window[1]+crop_window[3] + 1, crop_window[0]:crop_window[0]+crop_window[2] + 1, :]
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    # input_image = input_image / 255.0
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    input_mask = input_mask / 255

    return input_image, input_mask

# U-net
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation="relu",
                  padding="same",
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation="relu",
                  padding="same",
                  # set 'kernel_initializer' same as above
                  kernel_initializer='he_normal')(conv)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
         ### START CODE HERE
        conv = Dropout(rate=dropout_prob)(conv)
         ### END CODE HERE


    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        ### START CODE HERE
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
        ### END CODE HERE

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """

    ### START CODE HERE
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding="same")(expansive_input)

    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation="relu",
                 padding="same",
                  # set 'kernel_initializer' same as above
                 kernel_initializer="he_normal")(conv)
    ### END CODE HERE

    return conv

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    ### START CODE HERE
    cblock1 = conv_block(inputs, n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block.
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3) # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)
    ### END CODE HERE

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ### START CODE HERE
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)
    ### END CODE HERE

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 # set 'kernel_initializer' same as above exercises
                 kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    ### START CODE HERE
    conv10 = Conv2D(n_classes, 1, padding="same")(conv9)
    ### END CODE HERE

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


def main():
    root_path = "./archivedDataset/wholefood"

    train_image_list, train_mask_list, train_bounding_list, dev_image_list, dev_mask_list, dev_bounding_list, test_image_list, test_mask_list, test_bounding_list = load_and_crop_training_data(root_path)

    train_dataset = build_dataset(train_image_list, train_mask_list, train_bounding_list)
    dev_dataset = build_dataset(dev_image_list, dev_mask_list, dev_bounding_list)
    test_dataset = build_dataset(test_image_list, test_mask_list, test_bounding_list)

    img_height = 96
    img_width = 128
    num_channels = 3

    train_ds = train_dataset.map(process_path)
    processed_train_ds = train_ds.map(preprocess)
    dev_ds = dev_dataset.map(process_path)
    processed_dev_ds = dev_ds.map(preprocess)
    test_ds = test_dataset.map(process_path)
    processed_test_ds = test_ds.map(preprocess)

    unet = unet_model((img_height, img_width, num_channels))

    unet.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    EPOCHS = 5
    VAL_SUBSPLITS = 5
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    train_dataset = processed_train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    validation_dataset = processed_dev_ds.cache().batch(BATCH_SIZE)
    test_dataset = processed_test_ds.cache().batch(BATCH_SIZE)
    print(processed_train_ds.element_spec)
    model_history = unet.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    test_loss, test_acc = unet.evaluate(test_dataset)
    print("TEST RESULTS:", test_loss, test_acc)

    unet.save("saved_unet_model")

if __name__ == "__main__":
    main()