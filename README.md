# WaferMap Defect Prediction

## Project Description
WaferMap Defect Prediction is an advanced machine learning project designed to analyze and classify wafer map defects based on their patterns. This project leverages image and numerical data, combining deep learning and preprocessing techniques to accurately predict defect types. The tool is tailored for applications in semiconductor manufacturing, ensuring streamlined defect detection and improved quality control.

---

## Key Features

- **Comprehensive Data Analysis**:
  - Preprocessing and transformation of both numerical and image data.
  - Handling missing values and standardizing features for optimal performance.

- **Deep Learning Model**:
  - A hybrid model combining CNN for image data and dense layers for numerical data.
  - Integration of numerical and image inputs for improved defect prediction accuracy.

- **Visualization**:
  - Distribution and frequency visualizations for wafer index and defect types.
  - Image previews of wafer maps and their associated defect classifications.

- **Scalability**:
  - Deployed using Streamlit, providing an interactive, web-based interface for real-time defect analysis.

---

## Technology Stack

- **Programming Languages**: Python
- **Libraries and Frameworks**:
  - **TensorFlow/Keras**: For designing and training the CNN model.
  - **Scikit-learn**: For preprocessing and data transformation.
  - **OpenCV & Matplotlib**: For image handling and visualization.
  - **Streamlit**: For web-based deployment.
  - **NumPy & Pandas**: For numerical computations and data manipulation.
  - **Ast & Pickle**: For data parsing and saving.

---

## Technical Overview

1. **Data Preprocessing**:
   - Converted categorical labels (e.g., defect types) to numerical values.
   - Applied `SimpleImputer` and `StandardScaler` to handle missing values and standardize numerical features.
   - Processed wafer map images using resizing and normalization techniques.

2. **Model Architecture**:
   - **CNN Branch**:
     - Extracted features from wafer map images using multiple convolutional and pooling layers.
   - **Dense Branch**:
     - Processed numerical inputs through dense layers with ReLU activation.
   - **Combined Model**:
     - Merged CNN and dense branches using concatenation, followed by dense layers and a softmax output layer for multi-class classification.

3. **Training and Evaluation**:
   - Utilized early stopping and model checkpoints to prevent overfitting.
   - Achieved detailed evaluation using confusion matrix and classification report.

4. **Visualization**:
   - Plotted wafer index distributions and defect type frequencies.
   - Displayed sample wafer maps with their predicted defect classifications.

---

## Deployment

The project is deployed using **Streamlit**, enabling an interactive and user-friendly web interface for defect analysis. The deployment provides the following features:

- Input options for wafer maps and numerical data.
- Real-time predictions of defect types based on user inputs.
- Visualizations for analyzed wafer maps and their respective classifications.

## Conclusion
WaferMap Defect Prediction demonstrates the power of combining deep learning with numerical data preprocessing to deliver an efficient and accurate defect classification tool. Designed for scalability and real-world applications, this project ensures reliable predictions, making it a valuable asset in quality assurance processes within semiconductor manufacturing. The deployment via Streamlit further enhances accessibility, allowing users to seamlessly interact with the model and gain insights in real time.

