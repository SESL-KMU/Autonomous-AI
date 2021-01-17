from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=255,  input_height=192, input_width=320  )
# model.load_weights('./checkpoints/color/color.')

model.train(
    train_images =  "./data/train/",
    train_annotations = "./data/train_color/",
    checkpoints_path = "./checkpoints/color/color" , epochs=30
)
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("./data/val/b1c66a42-6f7d68ca.jpg")
cv2_imshow(img)
out = model.predict_segmentation(
    inp="./data/val/b1c66a42-6f7d68ca.jpg",
    out_fname="./output.png"
)

cv2_imshow(out)
# import matplotlib.pyplot as plt
# plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="./data/val/"  , annotations_dir="./data/val_color/" ) )


# python -m keras_segmentation train --checkpoints_path="../../checkpoints" --train_images="../../train/" --train_annotations="../../train_aug/" --val_images="../../val/" --val_annotations="../../val_aug/" --n_classes=3 --input_height=192 --input_width=320 --model_name="vgg_unet"