# Semantic Segmentation for Autonomous Vehicles using U-Net

## 📌 Project Overview

This project focuses on real-time semantic segmentation for autonomous vehicles using the U-Net deep learning architecture. The model is trained to distinguish between different parts of road scenes (e.g., road, vehicles, pedestrians) and is deployed through a Flask web application to provide interactive inference capabilities.

---

## 🧠 U-Net Architecture

U-Net is a convolutional neural network designed for semantic segmentation. It consists of:

- **Encoder**: Extracts features using convolution and max-pooling layers
- **Bottleneck**: Deepest layer capturing context
- **Decoder**: Upsamples and refines feature maps using transposed convolutions and skip connections

📌 U-Net uses skip connections from encoder to decoder to retain spatial information.

---

## 🗃 Dataset Used

We used the dataset from the **Lyft-Udacity Self-Driving Challenge**.

📁 Dataset Folder Structure:

```
/kaggle/input/lyft-udacity-challenge/
 └── {data}/
     ├── CameraRGB/  # Input images
     └── CameraSeg/  # Ground truth segmentation masks
```




## 🧪 Model Training

1. **Preprocessing**

   - Resized images to (128x128)
   - Normalized pixel values
   - Converted masks to categorical class labels

2. **Model Compilation**

   - Loss Function: `categorical_crossentropy`
   - Optimizer: `Adam`
   - Metrics: `accuracy`

3. **Training Details**

   - Trained for 25+ epochs
   - Batch size: 16
   - Used early stopping and model checkpointing

📁 Output model saved as:

```bash
unet_model.h5
```

---

## 🌐 Flask Deployment

The trained model is deployed with Flask to allow real-time predictions via a web UI.

### Features:

- Upload input image
- Segment road scene
- Display output mask

### Flask File Structure:

```
app.py                # Flask app
static/uploads/       # User-uploaded images
static/results/       # Segmented output
templates/index.html  # HTML frontend
```

### How to Run:

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🧩 Technologies Used

- Python, NumPy, Pandas
- TensorFlow, Keras
- OpenCV, PIL
- Flask, HTML/CSS/Bootstrap
- Jupyter Notebooks

---

## 🚀 Future Improvements

- Add support for real-time video segmentation
- Integrate with front-end frameworks like React
- Deploy on cloud (Heroku, AWS, Render)



## 🙌 Credits

Dataset: Lyft x Udacity Challenge

---

## 📬 Contact

For questions or collaborations, reach out via [LinkedIn]([https://www.linkedin.com](https://www.linkedin.com/in/kandregula-prem-kumar-059642238)) 

