from dataset import Dataset
from models.net_1.model import Model

import numpy

from PIL import Image, ImageDraw

root_path = "/Users/michal/dataset/background/"
#root_path = "/home/michal/dataset/background/"

def show(input, output):
    image = Image.fromarray(input[0]*255)
    img1 = ImageDraw.Draw(image)   

    x0 = output[0]
    y0 = image.height
    x1 = output[1]
    y1 = image.height//2

    shape = [(x0, y0), (x1, y1)] 
    img1.line(shape, fill ="red", width = 4) 

    image.show()


size = 96
dataset = Dataset(size, size, 100, 100, root_path)


model = Model(dataset.training.input_shape, dataset.training.output_shape[0])

model.load("models/net_1/")

batch_size = 10



input, target = dataset.testing.get_batch()

prediction = model.forward(input)


for i in range(batch_size):
    input_np        = input[i].detach().numpy()
    target_np       = numpy.clip((target[i].detach().numpy() + 1.0)/2.0, 0, 1)
    prediction_np   = numpy.clip((prediction[i].detach().numpy() + 1.0)/2.0, 0, 1)

    target_np       = numpy.round(size*target_np)
    prediction_np   = numpy.round(size*prediction_np)

    print(target_np, prediction_np)
 
    show(input_np, prediction_np)