from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=3,  input_height=192, input_width=320  )
model.load_weights('./checkpoints/7000/7000.12')

model.train(
    train_images =  "./data/train/",
    train_annotations = "./data/train_aug/",
    checkpoints_path = "./checkpoints/70000/70000" , epochs=30
)
import cv2

img = cv2.imread("./data_origin/val/b1c66a42-6f7d68ca.jpg")
cv2.imshow("img", img)
out = model.predict_segmentation(
    inp="./data_origin/val/b1c66a42-6f7d68ca.jpg",
    out_fname="./output.png"
)

cv2.imshow("out", out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="./data/val/"  , annotations_dir="./data/val_aug/" ) )


