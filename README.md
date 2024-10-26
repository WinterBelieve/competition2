**Chinese Handwriting Recognition (Chinese OCR)**

Welcome to the Chinese Handwriting Recognition project! This project focuses on recognizing traditional Chinese handwritten characters using deep learning techniques.

**Table of Contents**

- [Project Background](#project-background)
- [Dataset Description](#dataset-description)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Contact Information](#contact-information)

**Project Background**

Handwritten Chinese character recognition is a challenging task due to the large number of characters and the variability in handwriting styles. This project aims to build a convolutional neural network (CNN) model capable of recognizing 4,803 traditional Chinese characters from handwritten images.

**Dataset Description**

**Dataset Location**

The dataset is available in this repository under the directory:

`Traditional-Chinese-Handwriting-Dataset/data/cleaned\_data(50\_50)/`

**Dataset Structure**

The dataset is organized into folders, each representing a single Chinese character. Each folder contains images of handwritten samples of that character.

Example directory structure:

` Traditional-Chinese-Handwriting-Dataset/ `
` └── data/`
`    └── cleaned_data(50_50)/`
`        ├── 丁/`
`        │   ├── img1.png`
`        │   ├── img2.png`
`        │   └── ...`
`        ├── 七/`
`        │   ├── img1.png`
`        │   ├── img2.png`
`        │   └── ...`
`        └── ... (4803 folders in total)`

- **Total Characters**: 4,803
- **Image Format**: PNG
- **Image Size**: 64x64 pixels (can be adjusted as needed)

**Installation Guide**

1. **Clone the Repository**
` git clone https://github.com/WinterBelieve/competition2.git `
` cd competition2 `


1. **Ensure Required Libraries Are Installed**

Make sure you have the necessary libraries installed (e.g., PyTorch, torchvision). It's assumed you have these set up in your environment.

1. **Set Up the Dataset**

The dataset should already be in place within the repository. If not, ensure that the dataset directory is correctly placed as per the structure mentioned above.

**Usage Instructions**

**Training the Model**

To train the model, simply run:

` python train.py `

- The train.py script will begin training the CNN model using the dataset provided.
- You can modify training parameters like epochs, batch size, or learning rate directly in the train.py script if needed.

**Testing the Model**

To evaluate the model's performance:

` python test.py `

- The test.py script will run the trained model on the test dataset and output the accuracy.
- Ensure that the trained model weights are saved and loaded correctly in the script.

**Project Structure**

` competition2/ `
` ├── train.py                                # Training script`
`├── test.py                                 # Testing/Evaluation script`
`├── model.py                                # Model definition`
`├── utils.py                                # Utility functions (if any)`
`├── Traditional-Chinese-Handwriting-Dataset/`
`│   └── data/`
`│       └── cleaned\_data(50\_50)/            # Dataset directory`
`│           ├── 丁/`
`│           ├── 七/`
`│           └── ... (4803 character folders)`
`└── README.md                               # Project README file`

- **train.py**: Script to train the CNN model.
- **test.py**: Script to test the trained model.
- **model.py**: Contains the CNN model architecture.
- **utils.py**: Contains utility functions used across scripts.
- **Traditional-Chinese-Handwriting-Dataset/**: Directory containing the dataset.

**Contact Information**

For any questions or issues, please contact:

- **Email**: d1300701@cgu.edu.tw

