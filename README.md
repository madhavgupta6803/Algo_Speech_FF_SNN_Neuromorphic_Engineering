# Algo_Speech_FF_SNN_Neuromorphic_Engineering

There are two files in this project
### SNN_Without_Delay_SHD_Dataset.ipynb:

#### The notebook implements a spiking neural network (SNN) without delay, designed for processing event-based data from the Spiking Heidelberg Dataset (SHD). Here are the primary functionalities and steps:

#### Setup and Libraries:
Installs and imports the required libraries, snntorch and tonic.
Sets up device configuration for using GPU if available.

#### Data Preprocessing with Rasterization:
Defines _SHD2Raster class to convert event-based SHD data into a frame format that is compatible with neural networks.
Configures constants and transformation pipelines, including temporal and spatial downsampling, using tonic transformations.

#### Data Loading:
Loads the SHD dataset using the tonic.datasets.SHD class, applies transformations, and creates DataLoaders for training and testing data.

#### Network Definition:
Defines a spiking neural network model (Net) using PyTorch and snntorch.
The model architecture includes three fully connected layers with Leaky Integrate-and-Fire (LIF) neuron models in each layer.
The forward method processes input sequences through the network layers over time, updating membrane potentials and capturing spiking outputs.

#### Training and Testing Setup (assuming additional cells):
Initializes the model on the appropriate device (CPU/GPU) and counts the total parameters in the network.

### SNN_With_Delay_and_Adaptive_Scheduler.ipynb:

## How to run the files
### Dependencies Installation: 
Installing required libraries (snntorch and tonic).

### Environment Configuration: 
Initializing the PyTorch device (CPU/GPU).

### SHD Dataset Processing: 
A class (_SHD2Raster) is used to preprocess the Spiking Heidelberg Dataset (SHD) for compatibility with spiking neural networks (SNN).
