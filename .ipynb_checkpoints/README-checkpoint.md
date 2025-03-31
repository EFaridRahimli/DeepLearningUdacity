# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

![Dataset Structure](./viz.png)
### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
In order to gurantee better training. Following shortcomings are adressed in debugger rules
- Vanishing Gradients
- Overfitting
- Overtraining
- Poor Weights


### Results
Debugger provided healthy loss curve with rapid decrease in the initial phase as well as decreasing throught the training.

In order to speed up training GPU instance type was selected. Profiler showed helathy logs for both CPU and GPU logs.
CPU visualization in the notebook showed high utilization at the start which is correct. CPU was utilized to load the data. Concurrently GPU was not utilized.
GPU was peaking during the training and at 0 during other segments of training.

![Tuning1](screenshots/tuning1.JPG)
![Tuning2](screenshots/tunin2.JPG)
![Tuning3](screenshots/tuning3.JPG)
![cpuUtilization](screenshots/cpu.png)
![gpuUtilization](screenshots/gpu.png)
![loss](screenshots/loss.png)

PDF can be found in the project directory as Debugger Profiler Report.pdf


## Model Deployment
Final trained model was deployed to the Sagemaker as an inference endpoint used with the PyTorchModel. Model was deployed on the "ml.m5.large" instance type as not a lot of predictions are expected.

Endpoint accepts the data as raw image bytes therefore all inputs have to be transformed.

Predictions are returned as JSON therfore, have to parsed.

Lastly predictions are shown as index therfore, have to be mapped to appropriate class.

![endpoint](screenshots/endpoint.JPG)
