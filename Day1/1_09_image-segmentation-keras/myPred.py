from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=3,  input_height=192, input_width=320  )
model.load_weights('./checkpoints/70000/70000.6')

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("./data/val_/b1c66a42-6f7d68ca.jpg")
cv2_imshow(img)

h,w = img.shape[:2]
result = model.predict_segmentation(
    inp=img,
    out_fname="./output.png"
)
result = np.uint8(127 * result)
mask = cv2.resize(result,dsize=(w,h))
cv2_imshow(mask)


# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="./data/val_/"  , annotations_dir="./data/val_aug_/" ) )

