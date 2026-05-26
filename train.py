# Train the network with loging and saving
#from tqdm import tqdm
import setup
from setup import getdataloaders, Net
from setup import  gettestloaders
import json
import torch
import argparse
#import nn
from setup import loss_function
import torch.optim as optim



def main(args):
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}") # not yet being used
    print(f"Learning rate: {args.lr}")    #not yet being used
    print(f"Dataset path: {args.data_dir}") #not yet being used

    trainloader = getdataloaders(args.batch_size)
    testloader = gettestloaders(args.batch_size)

    net = Net()
    total_params = sum(p.numel() for p in net.parameters())
    print("Total Parameters:", total_params)
    
    #define optimiser
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0 #initialise best accuarcy
    logs = [] #initalise logs list

    #Training loops for epoch and batches
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0

    #    loop = tqdm(trainloader)

        #for i, data in enumerate(loop, 0):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # predict
            loss = loss_function(outputs, labels) # measure error
            loss.backward() # compute gradients
            optimizer.step()  # update weights

            # print statistics
            running_loss += loss.item()
#           loop.set_description(f"Epoch {epoch+1}")
#           loop.set_postfix(loss=loss.item())
        
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        #Evaluation each epoch
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy} % Total = {total}')

        #Checkpointing
        # Save latest (epoch) model checkpoint
        torch.save(
            net.state_dict(),
            f"checkpoints/last_model_epoch_{epoch+1}.pth"
        )

        print(f"Last model for epoch {epoch+1} saved.")

        # Save best model checkpoint
        if accuracy > best_acc:

            best_acc = accuracy

            torch.save(
                net.state_dict(),
                "checkpoints/best_model.pth"
            )

            print("Best model updated and saved.")

        #Experiment tracking
        logs.append({ 
        "epoch": epoch, 
        "loss": running_loss, 
        "accuracy": accuracy 
        }) 

        # Save Experiment tracking to file

        with open("metrics.json", "w") as f: 
            json.dump(logs, f, indent=4) 

    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training Script")

    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Dataset directory")

    args = parser.parse_args()

    main(args)
