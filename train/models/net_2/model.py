


import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ActivationQuantizer(nn.Module):
    def forward(self, input):
        range = 127.0
        return (torch.clamp(input, -1.0, 1.0)*range).round()/range

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width  = self.input_shape[2]    

        ratio           = 2**4 

        fc_inputs_count = ((input_width)//ratio)*((input_height)//ratio)
 
        self.layers = [ 
                        nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),
                        ActivationQuantizer(),
                        nn.ReLU(), 

                        nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
                        ActivationQuantizer(),
                        nn.ReLU(), 

                        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                        ActivationQuantizer(),
                        nn.ReLU(), 

                        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                        ActivationQuantizer(),
                        nn.ReLU(), 
                       
                        nn.Dropout(0.01),
                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                        ActivationQuantizer()
                    ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    def forward(self, state):
        return self.model.forward(state)
   
    def save(self, path):
        name = path + "trained/model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name, map_location = self.device))
        self.model.eval() 
    
