# Train the network with loging and saving and distributed training
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
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# Datalogging and checkpointing
from pathlib import Path
from datetime import datetime
import logging



def main(args):
    
    # Distributed setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Running DDP on GPU {local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        print(f"Running single device: {device}")
        
    # Print out training arg values        
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}") 
    print(f"Learning rate: {args.lr}")    
    print(f"Dataset path: {args.data_dir}") #not yet being used

    # Update dataloaders for DistributedSampler
    trainloader, trainsampler = getdataloaders(
        args.batch_size,
        distributed
    )
    testloader = gettestloaders(args.batch_size)

    #Move model to GPU
    net = Net().to(device)
    
    # Wrap model with DDP
    if distributed:
        net = DDP(net, device_ids=[local_rank])
        
    total_params = sum(p.numel() for p in net.parameters())
    print("Total Parameters:", total_params)

    
    #define optimiser
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # Create unique experiment timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment directories
    run_dir = Path(f"runs/run_{timestamp}")
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  

    # Configure logging
    logging.basicConfig(
        filename=run_dir / "training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Training started")     

    best_acc = 0 #initialise best accuarcy
    logs = [] #initalise logs list

    config = {
        "epochs": 2,
        "optimizer": optimizer.__class__.__name__,
        "loss_function": "CrossEntropyLoss",
        "timestamp": timestamp
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    #Training loops for epoch and batches
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        #Set epoch for sampler
        if distributed:
            trainsampler.set_epoch(epoch)
            
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0       

    #    loop = tqdm(trainloader)

        #for i, data in enumerate(loop, 0):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # predict
            loss = loss_function(outputs, labels) # measure error
            loss.backward() # compute gradients
            optimizer.step()  # update weights

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1                

        
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
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = epoch_loss / batch_count

        print(f'Accuracy of the network on the 10000 test images: {accuracy} % Total = {total}')
        logging.info(
            f"Epoch={epoch+1} "
            f"Loss={avg_loss:.4f} "
            f"Accuracy={accuracy:.2f}%"
        )        

        #Checkpointing
        # Save latest (epoch) model checkpoint
        # Save checkpoints only on rank 0, otherwise both GPUs try writing simultaneously.
        if local_rank == 0:
            
            torch.save(
                net.state_dict(),
                checkpoint_dir / f"last_model_epoch_{epoch+1}.pth"
            )

            print(f"Last model for epoch {epoch+1} saved.")
            logging.info(
                f"Epoch {epoch+1}: latest checkpoint saved"
            )
 
            if accuracy > best_acc:
                best_acc = accuracy
                
                torch.save(
                    net.state_dict(),
                    checkpoint_dir / f"best_model_{timestamp}.pth"
                )
                
                print("Best model updated and saved.")
                logging.info(
                    f"Epoch {epoch+1}: new best model saved (accuracy={accuracy:.2f})"
                )

        #Experiment tracking
        
        logs.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 2),
            "best_accuracy": round(best_acc, 2)
        })
        
        # Save Experiment tracking to file

        with open(run_dir / "metrics.json", "w") as f: 
            json.dump(logs, f, indent=4) 

    print('Finished Training')
    logging.info("Training completed")
    logging.info(f"Best accuracy: {best_acc:.2f}%")

    print(f"\nExperiment saved to: {run_dir}")
    print(f"Best accuracy achieved: {best_acc:.2f}%")

    if distributed:
        dist.destroy_process_group()

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
