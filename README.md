**Chinese Handwriting Recognition (Chinese OCR)中文手寫辨識**

Welcome to the Chinese Handwriting Recognition competition!  

歡迎來到中文手寫辨識競賽！  

This project focuses on recognizing traditional Chinese handwritten characters using deep learning techniques.  

此專案旨在使用深度學習技術來辨識繁體中文手寫。  

**Table of Contents 目錄**

- [Competition Background](#Competition-background)
- [Dataset Description](#dataset-description)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Contact Information](#contact-information)

**Competition Background 競賽背景**

Handwritten Chinese character recognition is a challenging task due to the large number of characters and the variability in handwriting styles.   
This project aims to build a convolutional neural network (CNN) model capable of recognizing 4,803 traditional Chinese characters from handwritten images.  

中文手寫辨識是一項具有挑戰性的任務，因為它涉及大量的字符以及多變的書寫風格。  
本競賽旨在建立一個卷積神經網絡（CNN）模型，能夠辨識從手寫圖片中提取出的 4,803 個繁體中文。  

**Dataset Description**
***Dataset Location***

The dataset is available in this repository under the directory:  
資料集已包含在本專案的以下路徑中：  

Traditional-Chinese-Handwriting-Dataset/data/cleaned\_data(50\_50)/

**Dataset Structure**

The dataset is organized into folders, each representing a single Chinese character. Each folder contains images of handwritten samples of that character.  
資料集以資料夾方式組織，每個資料夾代表一個中文字，其中包含該字的手寫樣本圖片。  

Example directory structure:  
資料夾結構範例：
```
Traditional-Chinese-Handwriting-Dataset/ 
└── data/
    └── cleaned_data(50_50)/
        ├── 丁/
        │   ├── 丁_0.png
        │   ├── 丁_1.png
        │   └── ...
        ├── 七/
        │   ├── 七_0.png
        │   ├── 七_1.png
        │   └── ...
        └── ... (4803 folders in total)
```

- **Total Characters 總字數**: 4,803
- **Image Format 圖片格式**: PNG
- **Image Size 圖片大小**: 64x64 pixels (can be adjusted as needed)

**Installation Guide 安裝指南**


1. **Clone the Repository**
` git clone https://github.com/WinterBelieve/competition2.git `
` cd competition2 `


2. **Ensure Required Libraries Are Installed 確認已安裝所需的函式庫**

Make sure you have the necessary libraries installed (e.g., PyTorch, torchvision). It's assumed you have these set up in your environment.  
請確認環境中已安裝所需函式庫（如 PyTorch、torchvision）。  
假設您已經在環境中完成安裝設定。

3. **Set Up the Dataset 設置資料集**

The dataset should already be in place within the repository. If not, ensure that the dataset directory is correctly placed as per the structure mentioned above.  
資料集應該已包含於此倉庫中。  
如果沒有，請依照上述結構將資料集放置於正確位置。


**Usage Instructions使用說明**

**Training the Model 訓練模型**

To train the model, simply run:  
執行以下指令以訓練模型：  

` python train.py `

- The train.py script will begin training the CNN model using the dataset provided.
- 此 train.py 腳本將使用所提供的資料集來訓練 CNN 模型。
- You can modify training parameters like epochs, batch size, or learning rate directly in the train.py script if needed.
- 你可以直接在 train.py 腳本中修改訓練參數，例如 epoch 次數、批次大小或學習率。


**Testing the Model 測試模型**

To evaluate the model's performance:  
執行以下指令來評估模型性能：  

` python evaluate.py `

- The test.py script will run the trained model on the test dataset and output the accuracy.
- evaluate.py 腳本將使用測試集評估訓練好的模型並輸出準確率。
- Ensure that the trained model weights are saved and loaded correctly in the script.
- 確保腳本中已正確載入訓練模型的權重。


**Competition Structure 競賽架構**

```
competition2/
├── train.py                                # Training script 訓練腳本
├── evaluate.py                             # Testing/Evaluation script 測試/評估腳本
├── Traditional-Chinese-Handwriting-Dataset/
│   └── data/
│       └── cleaned\_data(50\_50)/            # Dataset directory 資料集資料夾
│           ├── 丁/
│           ├── 七/
│           └── ... (4803 character folders)
└── README.md                               # README file 說明文件
```

- **train.py**: Script to train the CNN model. 訓練 CNN 模型的腳本
- **evaluate.py**: Script to test the trained model. 測試訓練模型的腳本
- **Traditional-Chinese-Handwriting-Dataset/**: Directory containing the dataset.

**Contact Information 聯絡資訊**

For any questions or issues, please contact:  
如有任何問題，請聯絡：

- **Email 電子郵件**: d1300701@cgu.edu.tw

