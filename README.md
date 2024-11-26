## Algo_Speech_FF_SNN_Neuromorphic_Engineering

There are two files in this project
### SNN_Without_Delay_SHD_Dataset.ipynb:

#### The notebook implements a spiking neural network (SNN) without delay, designed for processing event-based data from the Spiking Heidelberg Dataset (SHD). Here are the primary functionalities and steps:

#### Setup and Libraries:
* Installs and imports the required libraries, snntorch and tonic.
* Sets up device configuration for using GPU if available.

#### Data Preprocessing with Rasterization:
* Defines _SHD2Raster class to convert event-based SHD data into a frame format that is compatible with neural networks.
* Configures constants and transformation pipelines, including temporal and spatial downsampling, using tonic transformations.

#### Data Loading:
Loads the SHD dataset using the tonic.datasets.SHD class, applies transformations, and creates DataLoaders for training and testing data.

#### Network Definition:
* Defines a spiking neural network model (Net) using PyTorch and snntorch.
* The model architecture includes three fully connected layers with Leaky Integrate-and-Fire (LIF) neuron models in each layer.
* The forward method processes input sequences through the network layers over time, updating membrane potentials and capturing spiking outputs.

#### Training and Testing Setup:
Training and Testing Accuracy and Loss are plotted for the Basic SNN Network without the delays scheduler. The test accuracy after 50 epochs is 61%

![image](https://github.com/user-attachments/assets/486ccb7b-f126-407d-902c-c69cee823ddc)
![image](https://github.com/user-attachments/assets/c6069a33-26fd-4009-bd98-15313a98fdd7)



### SNN_With_Delay_and_Adaptive_Scheduler.ipynb:

#### This notebook implements a spiking neural network (SNN) that includes delay mechanisms and an adaptive delay scheduler. Here are the primary functionalities and steps:

#### Setup and Imports:
Installs necessary libraries (snntorch, tonic), and sets up the device for GPU use if available.

#### Data Preprocessing:
The _SHD2Raster class and transformation pipeline rasterize the event-based SHD data into frame format for neural network processing.

#### Data Loading and Batching:
* Loads and batches the SHD dataset with padding to standardize tensor shapes, using tonic for DataLoader setup.
* Shuffles data for training, using a custom shuffle function.

#### Adaptive Delay Scheduler:
* Implements the AdaptiveDelayScheduler class to dynamically adjust the delay cap for neurons based on neuron activity within a sliding delay window.
* This scheduler updates delay caps based on the proportion of neurons falling within a specified delay range, allowing adaptive timing adjustments for neurons during training.

#### Network Definition with Adaptive Delays:
* The NetWithAdaptiveDelays class defines a three-layer spiking neural network with learnable delays.
* Each layer is a fully connected layer followed by Leaky Integrate-and-Fire (LIF) neurons, and delays are applied to spike trains based on each neuron’s learned delay.
* The apply_delays_to_sequence method applies these delays to spikes, shifting each neuron’s spike sequence according to its individual delay.
* The forward method processes inputs across time steps, collecting spiking outputs at each layer and adapting membrane potentials accordingly.
* The update_delays method adjusts hidden layer delays using the adaptive delay scheduler.

#### Training and Testing Setup:
Training and Testing Accuracy and Loss are plotted for SNN Network with Adaptive Delay Scheduler. After each epoch the delays in the network are updated using the adaptive scheduling mechanism which improves its accuracy from 61% without adaptive scheduling to 73% with adaptive scheduling incorporated.

![image](https://github.com/user-attachments/assets/97223b8e-2af1-4ee6-a779-38cfd1709711)
![image](https://github.com/user-attachments/assets/29ccad90-affb-4ec0-9259-32ed84ecd523)

#### Reference
https://ieeexplore.ieee.org/abstract/document/10094768

