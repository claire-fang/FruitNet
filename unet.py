import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

# def convert_yolo_box_to_corners(x_center, y_center, box_w, box_h, img_w, img_h):
#     # Make sure image dimensions are float for multiplication
#     img_w  = tf.cast(img_w, tf.float32)
#     img_h  = tf.cast(img_h, tf.float32)

#     # Convert YOLO relative coordinates to pixel coordinates
#     x_center = x_center * img_w
#     y_center = y_center * img_h
#     box_w    = box_w  * img_w
#     box_h    = box_h * img_h

#     # Compute corners
#     x1 = x_center - box_w / 2
#     y1 = y_center - box_h / 2
#     x2 = x_center + box_w / 2
#     y2 = y_center + box_h / 2

#     # Bounds checking
#     zero = tf.constant(0, dtype=tf.float32)
#     x1 = tf.maximum(zero, x1)
#     y1 = tf.maximum(zero, y1)
#     x2 = tf.minimum(img_w, x2)
#     y2 = tf.minimum(img_h, y2)

#     # Crop window format: [offset_height, offset_width, target_height, target_width]
#     offset_height = tf.cast(y1, tf.int32)
#     offset_width  = tf.cast(x1, tf.int32)
#     target_height = tf.cast(y2 - y1, tf.int32)
#     target_width  = tf.cast(x2 - x1, tf.int32)

#     return [offset_height, offset_width, target_height, target_width]

def convert_yolo_box_to_corners(x_center, y_center, box_w, box_h, img_w, img_h):
    # 1. Cast dimensions to float
    img_w = tf.cast(img_w, tf.float32)
    img_h = tf.cast(img_h, tf.float32)

    # 2. Convert to pixels
    box_w_px = box_w * img_w
    box_h_px = box_h * img_h
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h

    # 3. Compute top-left corner
    x1 = x_center_px - (box_w_px / 2)
    y1 = y_center_px - (box_h_px / 2)

    # 4. SAFETY: Clip to image bounds (leave at least 1px space)
    x1 = tf.clip_by_value(x1, 0.0, img_w - 1.0)
    y1 = tf.clip_by_value(y1, 0.0, img_h - 1.0)

    # 5. SAFETY: Calculate max available size
    max_w = img_w - x1
    max_h = img_h - y1
    
    target_width = tf.minimum(box_w_px, max_w)
    target_height = tf.minimum(box_h_px, max_h)

    # 6. CRITICAL FIX: Force Minimum 1 pixel size to prevent crash
    target_width = tf.maximum(target_width, 1.0)
    target_height = tf.maximum(target_height, 1.0)

    return tf.cast(y1, tf.int32), tf.cast(x1, tf.int32), tf.cast(target_height, tf.int32), tf.cast(target_width, tf.int32)

def load_and_crop_training_data(root_path):
    # Read CSV
    csv_path = os.path.join(root_path, "wholefood_annotations.csv")
    df = pd.read_csv(csv_path)

    # Filter training rows
    train_df = df[(df["train_test_validation"] == 0) & (df["warm_color_binary"] == 1) & (df["mask_path"].notna())]
    print(train_df)

    image_list = []
    mask_list = []
    bounding_list = []

    for _, row in train_df.iterrows():
        image_name = row["file_name"]
        img_path = f"{root_path}/images/test/{row["image_path"]}"
        mask_path = f"{root_path}/masks/{image_name}/{row["mask_path"]}"

        image_list.append(img_path)
        mask_list.append(mask_path)

        bounding_list.append((row["x_center"], row["y_center"], row["width"], row["height"]))

        # h, w = img.shape[:2]

        # # ---- Convert YOLO bbox to pixel coordinates ----
        # x1, y1, x2, y2 = convert_yolo_box_to_corners(row["x_center"], row["y_center"], row["width"], row["height"], w, h)

    #     # ---- Crop both image and mask ----
    #     cropped_img  = img[y1:y2, x1:x2]
    #     cropped_mask = mask[y1:y2, x1:x2]

    return image_list, mask_list, bounding_list


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

img_height = 96
img_width = 128
num_channels = 3

def process_path(image_path, mask_path, bounding_box):

    img = tf.io.read_file(image_path)
    img_shape = tf.image.extract_jpeg_shape(img)
    crop_window = convert_yolo_box_to_corners(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], img_shape[1], img_shape[0])
    img = tf.image.decode_jpeg(img, channels=3)
    img = img[crop_window[1]:crop_window[1]+crop_window[3] + 1, crop_window[0]:crop_window[0]+crop_window[2] + 1, :]
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)

    mask = mask[crop_window[1]:crop_window[1]+crop_window[3] + 1, crop_window[0]:crop_window[0]+crop_window[2] + 1, :]
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    return input_image, input_mask


if __name__ == "__main__":
    unet = unet_model((img_height, img_width, num_channels))
    unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    root_path = "./wholefood"
    
    image_list, mask_list, bounding_list = load_and_crop_training_data(root_path)

    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)
    boounding_boxes = tf.constant(bounding_list)
        
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames, boounding_boxes))
    
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    i = 0
    for img, mask in processed_image_ds.take(10):
    # Convert to NumPy arrays
        # img_np = img.numpy()
        # mask_np = mask.numpy()
        # print(np.sum(mask_np != np.zeros_like(mask_np)))  # Check for all-zero mask

        # # Plot
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(img_np)
        # ax[0].set_title("Cropped Image")
        # ax[0].axis('off')

        # ax[1].imshow(mask_np)  # 1-channel mask
        # ax[1].set_title("Cropped Mask")
        # ax[1].axis('off')

        # plt.show()
        img_np = img.numpy()
        mask_np = mask.numpy()
        
        count = np.sum(mask_np > 0)
        print(f"Image {i}: Mask pixel count = {count}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_np)
        ax[0].set_title("Cropped Image")
        ax[1].imshow(mask_np.squeeze(), cmap='gray')
        ax[1].set_title("Cropped Mask")
        
        # Save instead of show
        save_path = f"debug_crop_{i}.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
        i += 1

    # EPOCHS = 5
    # VAL_SUBSPLITS = 5
    # BUFFER_SIZE = 500
    # BATCH_SIZE = 32
    # train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # print(processed_image_ds.element_spec)
    # model_history = unet.fit(train_dataset, epochs=EPOCHS)