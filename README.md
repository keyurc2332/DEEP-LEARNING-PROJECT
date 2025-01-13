# DEEP-LEARNING-PROJECT

**Company:** CODTECH IT SOLUTIONS PVT. LTD.

**Name:** KEYUR AMIT CHAUHAN

**Intern ID:** CT08EOQ

**Domain:** DATA SCIENCE

**Duration:** December 17th, 2024 - January 17th, 2025 (4 weeks)

**Mentor:** Neela Santhosh Kumar,HR & Academic Head

The deep learning project focused on building an image classification model using TensorFlow, leveraging the CIFAR-10 dataset, which contains 60,000 images across 10 categories, such as airplanes, cars, and birds.

Data Loading and Preprocessing:
The CIFAR-10 dataset was loaded directly from TensorFlow's datasets module. The images and labels were divided into training and testing sets. Preprocessing included normalizing pixel values to the range [0, 1] to enhance model performance and efficiency.

Model Architecture:
A convolutional neural network (CNN) was designed for image classification. The architecture included:

Convolutional Layers: Extracted spatial features using filters.
Max Pooling Layers: Reduced dimensionality while preserving important features.
Dense Layers: Flattened and connected features to the final output layer.
Activation Functions: Relu was used for hidden layers, and softmax for the output layer to classify images into 10 categories.
Model Compilation:
The model was compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric. This ensured an efficient and accurate learning process.

Model Training:
The model was trained for 10 epochs using the training dataset, with a batch size of 32. The training process involved monitoring the validation accuracy and loss to evaluate the model's generalization capabilities.

Evaluation and Testing:
The trained model was evaluated on the test dataset, achieving a test accuracy of approximately 71.6%. This demonstrated its ability to classify images from unseen data.

Results Visualization:

Confusion matrix and classification report were used to understand the performance on individual categories.
A grid of test images was displayed, showing predicted labels alongside their true labels, highlighting the model's performance visually.
Saving the Model:
The trained model was saved as an H5 file, enabling future reuse without retraining.

This project demonstrated the end-to-end implementation of a deep learning model, from data preprocessing and model building to evaluation and visualization. It effectively showcased the capability of CNNs in image classification tasks.
