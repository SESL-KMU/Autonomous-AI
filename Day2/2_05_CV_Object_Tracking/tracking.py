import cv2
import numpy as np

col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
RoI_hist = None

def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputmode
    global rectangle, RoI_hist, trackWindow

    if inputmode: #inputmode가 ON일떄 작동 (camShift코드에서 'i'버튼을 누르면 실행)
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
                cv2.imshow('frame', frame)

        elif event == cv2.EVENT_LBUTTONUP:
            inputmode = False
            rectangle = False

            cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
            height, width = abs(row - y), abs(col - x)
            trackWindow = (col, row, width, height)

            RoI = frame[row:row+height, col:col + width]
            RoI = cv2.cvtColor(RoI, cv2.COLOR_BGR2HSV)
            RoI_hist = cv2.calcHist([RoI], [0, 1], None, [180, 256], [0, 180, 0, 256])

            cv2.normalize(RoI_hist, RoI_hist, 0, 255, cv2.NORM_MINMAX)
            print(trackWindow)

    return

def camShift():

    global frame, frame2, inputmode, trackWindow, RoI_hist

    try:
        cap = cv2.VideoCapture("./KITTI_data.mp4")
    except Exception as e:
        print(e)
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./output.avi', fourcc, 30.0, (int(width), int(height)))

    # trackWindow = (356, 119, 357, 524)
    # RoI = frame[trackWindow[1]:trackWindow[1] + trackWindow[3], trackWindow[0]:trackWindow[0] + trackWindow[2]]
    # RoI = cv2.cvtColor(RoI, cv2.COLOR_BGR2HSV)
    # RoI_hist = cv2.calcHist([RoI], [0], None, [180], [0, 180])
    #
    # cv2.normalize(RoI_hist, RoI_hist, 0, 255, cv2.NORM_MINMAX)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if trackWindow is not None:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0, 1], RoI_hist, [0,180, 0, 256], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)

            x, y, w, h = trackWindow
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

        cv2.imshow('frame', frame)
        out.write(frame)
        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        if k == ord('i'):
            print("Select Area for CamShift and Enter a key")
            inputmode = True
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


camShift()