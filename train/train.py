from dataset import Dataset
from models.net_0.model import Model


dataset = Dataset(96, 96, 1000, 1000)


model = Model(dataset.training.input_shape, dataset.training.outputs_count)

import torch

epoch_count = 20
learning_rates = [0.0001, 0.0001, 0.0001, 0.0001, 0.00001]

testing_loss_sum_best = None

for epoch in range(epoch_count):
    
    batch_size  = 32
    batch_count = (dataset.training.get_count()+batch_size) // batch_size

    learning_rate = learning_rates[epoch%len(learning_rates)]
    
    optimizer  = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay=10**-6)  

    training_loss_sum = 0.0
    for batch_id in range(batch_count):
        training_x, training_y = dataset.training.get_batch(batch_size)

        training_x = training_x.to(model.device)
        training_y = training_y.to(model.device)

        predicted_y = model.forward(training_x)

        error = (training_y - predicted_y)**2
        loss  = error.mean() 

        loss.backward()
        optimizer.step()

        training_loss_sum+= loss.detach().to("cpu").numpy()

    training_loss_sum = training_loss_sum/batch_count

    
    batch_count = (dataset.testing.get_count()+batch_size) // batch_size
    testing_loss_sum = 0.0
    for batch_id in range(batch_count):
        testing_x, testing_y = dataset.testing.get_batch(batch_size)

        testing_x = testing_x.to(model.device)
        testing_y = testing_y.to(model.device)

        predicted_y = model.forward(testing_x)

        error = (testing_y - predicted_y)**2
        loss  = error.mean()

        testing_loss_sum+= loss.detach().to("cpu").numpy()

    testing_loss_sum = testing_loss_sum/batch_count


    print("epoch = ", epoch, training_loss_sum, testing_loss_sum)
    
    save_model = False
    
    if testing_loss_sum_best == None:
        testing_loss_sum_best = testing_loss_sum
        save_model = True

    if testing_loss_sum < testing_loss_sum_best:
        testing_loss_sum_best = testing_loss_sum
        save_model = True


    if save_model:
        model.save("models/net_0/")

        print("\n\n\n")
        print("=================================================")
        print("new best net in ", epoch, "\n")
        print("TRAINING result ")
        print(training_loss_sum)
        print("TESTING result ")
        print(testing_loss_sum)
        print("\n\n\n")