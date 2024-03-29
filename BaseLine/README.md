# CNN And MSB
We leverage the ASCAD Model for the purpose of the CNN study
| Layer            | Hyperparameters                                                    |
|------------------|--------------------------------------------------------------------|
| Conv 1           | filters: 64; kernel size: 11; stride: 2; Relu                     |
| Conv 2           | filters: 128; kernel size: 11; stride: 2; Relu                    |
| Conv 3           | filters: 256; kernel size: 11; stride: 2; Relu                    |
| Conv 4~5         | filters: 512; kernel size: 11; stride: 2; Relu                    |
| AvgPool 1~5      | pooling size: 2; stride: 2                                        |
| Dense 1~2        | No. of neurons: 4096; Relu                                        |
| Output           | No. of neurons: {9} (HW) or 256 (ID)/ {2} for MSB; Softmax                     |


# MLP
| Layer            | Hyperparameters                    |
|------------------|------------------------------------|
| Dense 1~5        | No. of neurons: 200; ReLU          |
| Output           | No. of neurons: {9} (HW) or 256 (ID); Softmax |


We Train and validate the model using 140k traces, with a train val split of 90:10 



To train a pruned deep learning model for HW/ID leakage detection, run the following command:
```python 
python train.py --input <input_data_file> --output <output_directory> --model_dir <model_directory> --network_type <network_type> --target_byte <target_byte>
```
# Arguments:

* input: The path to the input data file. The input data file should be in a NumPy .npy format.

* output: The path to the output directory. The output directory will contain the trained model and the training logs.

* model_dir: The path to the model directory. The model directory will contain the pre-trained model that will be used as the starting point for pruning.

* network_type: The type of network to train. The supported network types are hw_model and ID.

* target_byte: The target byte to detect leakage for. The target byte should be an integer in the range 0-255.
# Please note that this is the general training command and is vaild for all 3 baseline networks.
