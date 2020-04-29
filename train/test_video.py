import numpy
import cv2
import torch
from models.net_0.model import Model


width   = 96
height  = 96

model = Model((1, height, width), 2)
model.load("models/net_0/")


cap = cv2.VideoCapture(0)


input_width = 640
input_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH,input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,input_height)


def get_net_input(image):
        resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        input_np = numpy.array(resized)

        input_max = 255
        input_min = 0

        k = (1.0 - 0.0)/(input_max - input_min)
        q = 1.0 - k*input_max

        input_normalised = k*input_np + q

        input_normalised = numpy.clip(input_normalised, 0, 1)

        result = torch.zeros((1, 1, height, width))
        result[0][0] = torch.from_numpy(input_normalised)

        return result



while(True):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    input_t = get_net_input(img)

    prediction_t = model.forward(input_t)

    prediction_np = numpy.clip(prediction_t.detach().numpy(), -1.0, 1.0)
    position_x0 = int(input_width*(prediction_np[0][0] + 1.0)/2.0)
    position_x1 = int(input_height*(prediction_np[0][1] + 1.0)/2.0)


    x0 = position_x0
    y0 = 0

    x1 = position_x1
    y1 = input_height

    cv2.line(img, (x0, y0), (x1, y1), (255), 4)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


'''
def show(input, output):
    image = Image.fromarray((input[0] + 1.0)*127)
    img1 = ImageDraw.Draw(image)   

    shape = [(output[0], 0), (output[1], image.height)] 
    img1.line(shape, fill ="red", width = 4) 

    image.show()


size = 96
dataset = Dataset(size, size, 100, 100)


model = Model(dataset.training.input_shape, dataset.training.outputs_count)

model.load("models/net_0/")

batch_size = 4



input, target = dataset.testing.get_batch()

prediction = model.forward(input)


for i in range(batch_size):
    input_np        = input[i].detach().numpy()
    target_np       = target[i].detach().numpy()
    prediction_np   = prediction[i].detach().numpy()

    target_np       = numpy.round(size*target_np, 1)
    prediction_np   = numpy.round(size*prediction_np, 1)

    print(target_np)
    print(prediction_np)
    print("\n\n\n")

    show(input_np, prediction_np)
'''