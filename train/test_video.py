import numpy
import cv2
import torch
import time
from models.net_1.model import Model


model_input_width  = 96*4
model_input_height = 96*4

model = Model((1, model_input_height, model_input_width), 2)
model.load("models/net_1/")


cap = cv2.VideoCapture(0)


input_width        = 640
input_height       = 480



cap.set(cv2.CAP_PROP_FRAME_WIDTH,input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,input_height)

def get_prediction(frame):
    frame_grayscale   = numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_resized     = cv2.resize(frame_grayscale, (model_input_width, model_input_height), interpolation = cv2.INTER_AREA)
    
    frame_normalised = numpy.clip(frame_resized/255.0, 0, 1)

    frame_tensor = torch.zeros((1, 1, model_input_height, model_input_width)).to(model.device)
    frame_tensor[0][0] = torch.from_numpy(frame_normalised).to(model.device)

    prediction_tensor  = model.forward(frame_tensor)

    prediction_np      = numpy.clip(prediction_tensor[0][0].detach().to("cpu").numpy(), 0, 1)


    return prediction_np


fps = 0.0

frame_count = 0

while(True):
    ret, frame      = cap.read()

    time_start = time.time()
    prediction_np   = get_prediction(frame)
    time_stop = time.time()

    prediction_np   = cv2.resize(prediction_np, (input_width, input_height), interpolation = cv2.INTER_LANCZOS4)

    b, g, r = cv2.split(frame)

    r = r/255.0
    g = g/255.0
    b = b/255.0

    g = 0.5*g  + 0.5*prediction_np

    r = (r*255).astype(dtype=numpy.uint8)
    g = (g*255).astype(dtype=numpy.uint8)
    b = (b*255).astype(dtype=numpy.uint8)

    result = cv2.merge((b,g,r))


    '''
    frame_count+= 1
    if frame_count%20 == 0:
        cv2.imwrite("images/frame_" + str(frame_count) + ".jpg", result)
    '''

    cv2.imshow('frame', result)

    fps = 0.9*fps + 0.1*1.0/(time_stop - time_start)

    print("FPS = ", round(fps, 1))



    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

