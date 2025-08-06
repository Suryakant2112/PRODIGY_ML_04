# Real-Time Hand Gesture Recognition 

This project is a real-time hand gesture recognition system that uses a webcam to identify gestures like "fist," "palm," and "ok." It leverages Google's MediaPipe for high-fidelity hand landmark detection and a custom-trained PyTorch neural network for classification.

The model processes landmark data instead of raw image pixels, making it lightweight and fast.

\![Demo GIF of the hand gesture recognition system in action]
*(You can replace the line above with a GIF of your project working)*

-----

## Tech Stack 🛠️

  * **Python 3.11**
  * **PyTorch:** For building and training the neural network.
  * **OpenCV:** For webcam access and image rendering.
  * **MediaPipe:** For real-time hand tracking and landmark extraction.
  * **Pandas & Scikit-learn:** For data handling and preparation.

-----

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Prerequisites

Ensure you have **Python 3.11** installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Set Up the Virtual Environment and Install Dependencies

These instructions are based on the commands you provided.

  * **Create the Virtual Environment**
    ```bash
    # This command creates a virtual environment named 'gesture-env-3.11' using Python 3.11
    py -3.11 -m venv gesture-env-3.11
    ```
  * **Activate the Environment (Windows PowerShell)**
    > **Note:** The `Set-ExecutionPolicy` command is only needed once per session on Windows to allow the activate script to run.
    ```powershell
    # Allow script execution for the current process
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

    # Activate the virtual environment
    .\gesture-env-3.11\Scripts\activate
    ```
  * **Install Required Libraries**
    ```bash
    pip install torch numpy opencv-python mediapipe pandas scikit-learn
    ```
    > **Pro Tip:** You can also create a `requirements.txt` file by running `pip freeze > requirements.txt` and then install from it using `pip install -r requirements.txt`.

-----

## Usage

The project is run in three distinct steps:

### Step 1: Collect Gesture Data

Run the data collection script. This will open your webcam. Show a gesture and press the corresponding key to save the landmark data.

  * **f:** Fist
  * **p:** Palm
  * **o:** OK sign

Collect at least 50-100 samples per gesture for better accuracy. Press **'q'** to quit.

```bash
python collect_data.py
```

### Step 2: Train the Neural Network

Run the training script. It will read the `gestures.csv` file you just created, train the model, and save the trained weights and labels to `gesture_model.pth`.

```bash
python train_model.py
```

### Step 3: Run Real-Time Inference

This is the final step. Run the inference script to see your trained model recognize gestures in real time\!

```bash
python run_inference.py
```

-----

## Project Structure

```
.
├── gesture-env-3.11/   # Virtual environment folder
├── collect_data.py     # Script to capture and save gesture data
├── train_model.py      # Script to train the PyTorch model
├── run_inference.py    # Script to run the live gesture recognition
├── gestures.csv        # (Generated) The collected landmark data
└── gesture_model.pth   # (Generated) The trained model weights
```

-----

## Future Improvements 💡

  * Add a wider variety of gestures.
  * Train on a more diverse and larger dataset.
  * Implement recognition for dynamic gestures (e.g., swiping).
  * Build a simple UI to map gestures to specific computer actions.
