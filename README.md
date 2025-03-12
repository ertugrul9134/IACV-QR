# IACV-QR
Morfolojik filtreler

# Brain Tumor Detection from MRI Scans Using Morphological Filters

This project focuses on detecting brain tumors from MRI scans using **morphological filters** for preprocessing and deep learning for classification. Morphological operations (such as erosion, dilation, opening, and closing) are applied to enhance the quality of MRI scans, making it easier to identify tumor regions. The dataset used is the **Brain Tumor MRI Dataset** from Kaggle, which contains MRI images categorized into different types of brain tumors.

---

## Steps Performed

### 1. **Dataset Download**
   - The dataset was downloaded using the **Kaggle Hub API**.
   - The dataset contains two main folders: `Training` and `Testing`, each with subfolders for different tumor types (e.g., `glioma`, `meningioma`, `pituitary`, `no_tumor`).

   ```python
   import kagglehub
   path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
   print("Path to dataset files:", path)
   ```

---

### 2. **Morphological Filtering**
   - Morphological operations are applied to preprocess the MRI scans. These operations help remove noise, fill holes, and enhance the structure of the tumor regions.
   - The following morphological operations are used:
     - **Closing**: Fills small holes and gaps in the tumor regions.
     - **Opening**: Removes small noise and smooths the boundaries of the tumor regions.
     - **Morphological Gradient**: Highlights the edges of the tumor regions.

   ```python
   import cv2
   import numpy as np

   # Load an MRI scan
   image = cv2.imread("mri_scan.jpg", cv2.IMREAD_GRAYSCALE)

   # Apply thresholding to create a binary image
   _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

   # Define a kernel for morphological operations
   kernel = np.ones((5, 5), np.uint8)

   # Apply morphological operations
   morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
   morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
   morphed = cv2.morphologyEx(morphed, cv2.MORPH_GRADIENT, kernel, iterations=1)
   ```

---

### 3. **Data Preprocessing**
   - The preprocessed images are resized and normalized to prepare them for training.
   - Data augmentation techniques (e.g., rotation, flipping, zooming) are applied to increase the diversity of the training data.

   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   # Define image size and batch size
   img_size = (128, 128)
   batch_size = 32

   # Data augmentation and preprocessing
   train_datagen = ImageDataGenerator(
       rescale=1.0 / 255.0,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       validation_split=0.2,
   )

   # Load training and validation data
   train_generator = train_datagen.flow_from_directory(
       train_path,
       target_size=img_size,
       batch_size=batch_size,
       class_mode="categorical",
       subset="training",
   )

   validation_generator = train_datagen.flow_from_directory(
       train_path,
       target_size=img_size,
       batch_size=batch_size,
       class_mode="categorical",
       subset="validation",
   )
   ```

---

### 4. **Model Building**
   - A **Convolutional Neural Network (CNN)** is built for tumor classification.
   - The model consists of convolutional layers, max-pooling layers, and fully connected layers.

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

   # Define the model
   model = Sequential([
       Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(64, (3, 3), activation="relu"),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(128, (3, 3), activation="relu"),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation="relu"),
       Dropout(0.5),
       Dense(len(classes), activation="softmax"),
   ])

   # Compile the model
   model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
   ```

---

### 5. **Model Training**
   - The model is trained on the preprocessed and augmented MRI scans.
   - Training progress is monitored using validation accuracy and loss.

   ```python
   # Train the model
   history = model.fit(
       train_generator,
       steps_per_epoch=train_generator.samples // batch_size,
       validation_data=validation_generator,
       validation_steps=validation_generator.samples // batch_size,
       epochs=10,
       verbose=1,
   )
   ```

---

### 6. **Model Evaluation**
   - The trained model is evaluated on the test dataset to measure its performance.
   - The test accuracy and loss are reported.

   ```python
   # Evaluate the model on the test data
   test_loss, test_accuracy = model.evaluate(test_generator)
   print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
   ```

---

### 7. **Visualization**
   - The results are visualized, including:
     - Training and validation accuracy/loss curves.
     - Sample MRI scans with predicted tumor regions.

   ```python
   import matplotlib.pyplot as plt

   # Plot training history
   plt.figure(figsize=(12, 5))
   plt.subplot(1, 2, 1)
   plt.plot(history.history["accuracy"], label="Training Accuracy")
   plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
   plt.title("Training and Validation Accuracy")
   plt.xlabel("Epoch")
   plt.ylabel("Accuracy")
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(history.history["loss"], label="Training Loss")
   plt.plot(history.history["val_loss"], label="Validation Loss")
   plt.title("Training and Validation Loss")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend()

   plt.show()
   ```

---

## Key Features
- **Morphological Filters**: Used to preprocess MRI scans and enhance tumor regions.
- **Deep Learning**: A CNN model is trained to classify MRI scans into tumor types.
- **Data Augmentation**: Applied to increase the diversity of the training data.
- **Visualization**: Training progress and results are visualized for better understanding.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using the Kaggle Hub API:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
   ```
4. Run the preprocessing, training, and evaluation scripts:
   ```bash
   python preprocess.py
   python train.py
   python evaluate.py
   ```

---

## Results
- The model achieves ** YET TO BE FOUND % accuracy** on the test dataset.
- Morphological filters significantly improve the quality of MRI scans, making it easier to detect tumor regions.

---

## Future Work
- Experiment with more advanced morphological operations.
- Use transfer learning with pre-trained models (e.g., ResNet, VGG) for better performance.ResNet can quickly train very deep neural network models and, by linking across layers, can prevent the gradient disappearance problem.  weight ve bias değerleri efektif bir şekilde güncellenemiyor ve sırasında yaratması gereken etkiden (optimal olarak düşünürsek) çok daha fazla etki yaratır ve öğrenme süreci verimini yitiriyor.
- Extend the project to segment tumor regions using U-Net or similar architectures.
