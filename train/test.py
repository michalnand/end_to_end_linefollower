from dataset import Dataset
from models.net_0.model import Model

import numpy

from PIL import Image, ImageDraw


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