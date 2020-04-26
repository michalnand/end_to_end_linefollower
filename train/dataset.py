import numpy
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import torch

class Dataset:
    def __init__(self, width, height, training_count, testing_count):
        self.training = LinesDataset(width, height, training_count)
        self.testing  = LinesDataset(width, height, testing_count)


class LinesDataset:
    def __init__(self, width, height, items_count):

        self.width      = width
        self.height     = height
        self.channels   = 1

        self.input_shape = (self.channels, self.height, self.width)

        self.outputs_count = 2

        self.input  = numpy.zeros((items_count, self.channels, self.height, self.width))
        self.target = numpy.zeros((items_count, self.outputs_count))

        self._generate(items_count) 

    def get_count(self):
        return len(self.input)

    def show_example(self, idx):
        input = self.input[idx]

        input = (input + 1.0)*127

        img = Image.fromarray(input[0])
        img.show()



    def get_batch(self, batch_size = 32):
        input_t  = torch.zeros((batch_size, self.channels, self.height, self.width))
        target_t = torch.zeros((batch_size, self.outputs_count))

        for i in range(batch_size):
            idx = numpy.random.randint(self.get_count())
            input_t[i]  = torch.from_numpy(self.input[idx])
            target_t[i] = torch.from_numpy(self.target[idx])


        return input_t, target_t

    def _generate(self, count):
        for i in range(count):
            self.input[i], self.target[i]   = self._create_item()



            

    def _create_item(self):
        input  = numpy.zeros((self.channels, self.height, self.width))
        target = numpy.zeros(self.outputs_count)

        lines = []

        max_lines_count = 1
        lines_count = numpy.random.randint(max_lines_count) + 1

        for i in range(lines_count):
            r = 0.05
            if numpy.random.randint(2) == 0:
                x0 = numpy.random.rand()
                y0 = numpy.random.rand()*r

                x1 = numpy.random.rand()
                y1 = numpy.random.rand()*r + (1.0 - r)
            else:
                x0 = numpy.random.rand()
                y0 = 0

                x1 = numpy.random.rand()
                y1 = 1.0
            
            lines.append((x0, y0, x1, y1))

            '''
            target[0] = x0
            target[1] = y0
            target[2] = x1
            target[3] = y1
            '''

            target[0] = x0
            target[1] = x1

        img = self._generate_line_image(lines)

        noise_level = 0.1

        pixels = numpy.array(img)
        noise  = numpy.random.rand(self.height, self.width)*255

        img_noised      = (1.0 - noise_level)*pixels + noise_level*noise


        k = (1.0 - (-1.0))/(img_noised.max() - img_noised.min())
        q = 1.0 - k*img_noised.max()

        input[0] = k*img_noised + q

        '''
        for i in range(max_lines_count):
            target[4*i + 0] = lines[i%len(lines)][0]
            target[4*i + 1] = lines[i%len(lines)][1]
            target[4*i + 2] = lines[i%len(lines)][2]
            target[4*i + 3] = lines[i%len(lines)][3]
        '''

        return input, target

    def _generate_line_image(self, lines):
        img = Image.new("L", (self.width, self.height)) 

        line_width = 8 + numpy.random.randint(8)

        img1 = ImageDraw.Draw(img)   

        for line in lines:
            shape = [(line[0]*self.width, line[1]*self.height), (line[2]*self.width, line[3]*self.height)] 
            img1.line(shape, fill ="white", width = line_width) 
 
        #random perspective
        param = numpy.random.randint(200)
        coeff = self._create_coeff(
                                    (0, 0),
                                    (img.width, 0),
                                    (img.width, img.height),
                                    (0, img.height),
                                    (-param, 0),
                                    (img.width + param, 0),
                                    (img.width, img.height),
                                    (0, img.height)
                                )

        #img = img.transform( (img.width, img.height), method=Image.PERSPECTIVE, data=coeff)

        #random color inversion
        if numpy.random.randint(2) == 1:
            img = ImageOps.invert(img)

        #random filter
        filter = numpy.random.randint(3)
        if filter == 0:
            img = img.filter(ImageFilter.BLUR)
        elif filter == 1:
            img = img.filter(ImageFilter.SMOOTH_MORE)
        else:
            img = img

        return img


    def _create_coeff(  self,
                        xyA1, xyA2, xyA3, xyA4,
                        xyB1, xyB2, xyB3, xyB4):

        A = numpy.array([
                [xyA1[0], xyA1[1], 1, 0, 0, 0, -xyB1[0] * xyA1[0], -xyB1[0] * xyA1[1]],
                [0, 0, 0, xyA1[0], xyA1[1], 1, -xyB1[1] * xyA1[0], -xyB1[1] * xyA1[1]],
                [xyA2[0], xyA2[1], 1, 0, 0, 0, -xyB2[0] * xyA2[0], -xyB2[0] * xyA2[1]],
                [0, 0, 0, xyA2[0], xyA2[1], 1, -xyB2[1] * xyA2[0], -xyB2[1] * xyA2[1]],
                [xyA3[0], xyA3[1], 1, 0, 0, 0, -xyB3[0] * xyA3[0], -xyB3[0] * xyA3[1]],
                [0, 0, 0, xyA3[0], xyA3[1], 1, -xyB3[1] * xyA3[0], -xyB3[1] * xyA3[1]],
                [xyA4[0], xyA4[1], 1, 0, 0, 0, -xyB4[0] * xyA4[0], -xyB4[0] * xyA4[1]],
                [0, 0, 0, xyA4[0], xyA4[1], 1, -xyB4[1] * xyA4[0], -xyB4[1] * xyA4[1]],
                ], dtype=numpy.float32)

        B = numpy.array([
                xyB1[0],
                xyB1[1],
                xyB2[0],
                xyB2[1],
                xyB3[0],
                xyB3[1],
                xyB4[0],
                xyB4[1],
                ], dtype=numpy.float32)
        return numpy.linalg.solve(A, B)


if __name__ == "__main__":
    dataset = LinesDataset(96, 96, 1000)

    input, target = dataset.get_batch()
    dataset.show_example(0)

