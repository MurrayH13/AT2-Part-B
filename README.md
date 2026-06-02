# Readme File for Part B

There is one Kaggle notebook from which this project was run "AT2 part2B".

The train file in Part B has been modified to run on 2 GPUs if available, through Kaggle.

A bash shell (run_distributed.sh) has been created to facilitate such a distrubuted run

The implementation of the project has been broken up into:
setup.py -- sets things ups for the training file this includes:
            seting random seed
            defining data/test loaders
            defining the model as a class, 
            defining loss function

train.py -- trains the model and carries out logging and checkpointing operations and allows for some cli paramenters
            The following arguments are currently implemented
                        epochs - number of epochs to run, defaul = 5
                        batch_size, default = 32
                        lr - learning rate, default = 0.001
          -- this train.py version also includes logging of runs with datestamps and checkpoints for the model at last epoch and best epoch 

            example usage
           python train.py --epochs 2 --batch-size 4
           sample output:
               Running single device: cuda
               Training for 2 epochs
               Batch size: 4
               Learning rate: 0.001
               Dataset path: ./data
               Total Parameters: 62006
               [1,  2000] loss: 2.170
               [1,  4000] loss: 1.789
               [1,  6000] loss: 1.656
               [1,  8000] loss: 1.556
               [1, 10000] loss: 1.519
               [1, 12000] loss: 1.435
               Accuracy of the network on the 10000 test images: 48.07 % Total = 10000
               Last model for epoch 1 saved.
               Best model updated and saved.
               [2,  2000] loss: 1.372
               [2,  4000] loss: 1.353
               [2,  6000] loss: 1.303
               [2,  8000] loss: 1.298
               [2, 10000] loss: 1.288
               [2, 12000] loss: 1.266
               Accuracy of the network on the 10000 test images: 56.26 % Total = 10000
               Last model for epoch 2 saved.
               Best model updated and saved.


               Finished Training

               Experiment saved to: runs/run_20260602_054532
               Best accuracy achieved: 56.26%
