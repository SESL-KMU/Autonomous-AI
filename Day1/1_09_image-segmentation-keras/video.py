from keras_segmentation.models.unet import vgg_unet
import cv2
from google.colab.patches import cv2_imshow

model = vgg_unet(n_classes=3,  input_height=192, input_width=320  )
model.load_weights('./checkpoints/7000/7000.12')


cap = cv2.VideoCapture('../0531_fps_20_short.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('./output.mp4', fourcc, 20.0, (854, 480))


while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        try:
            out = model.predict_segmentation(
                inp=frame
                # out_fname="./output.png"
            )
            cv2_imshow(out)

        except:
            continue

        out.write(out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

