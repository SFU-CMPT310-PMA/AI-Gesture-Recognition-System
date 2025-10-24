# AI-Gesture-Recognition-System

## Requires Python 3.9 - 3.11

- Python with MediaPipe is Compatible only with 3.9 - 3.12
- We tested and set up the environment with Python 3.11
- Check your Python version: `python --version`

## How to Run

1. Create a virtual environment: `python -m venv venv311` or to use Python version 3.11 `py -3.11 -m venv venv311` <br>
   **Note:** Make sure to install Python 3.11 or replace 3.11 with your version. You can change the name of the virtual environment `venv311` as you like.
2. Run the virtual environment: `.\venv311\Scripts\activate`
3. Install all the required libraries: `pip install -r requirements.txt`
4. Run the script: `python hand_tracking.py` to try it out!
5. Deactivate the virtual environment once you're done: `deactivate`

**NOTE**: Make sure your virtual environment (e.g. `venv311`) is included in your `.gitignore` if you are going to push to the repository. As default, `venv311` is already included in `.gitignore`.
