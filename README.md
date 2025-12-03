# AI-Gesture-Recognition-System

## Requires Python 3.9 - 3.11

- Python with MediaPipe is Compatible only with 3.9 - 3.12
- We tested and set up the environment with Python 3.11
- Check your Python version: `python --version`

## How to Run
You MUST be in the project’s root directory to run the program as being in other directories (like backend or legacy) will mess with the relative pathing used for loading models.
1. Create a virtual environment: `python -m venv venv311` or to use Python version 3.11 `py -3.11 -m venv venv311` <br>
   **Note:** Make sure to install Python 3.11 or replace 3.11 with your version. You can change the name of the virtual environment `venv311` as you like.
2. Run the virtual environment: `.\venv311\Scripts\activate` or `source /venv311/bin/activate`
3. Install all the required libraries: `pip install -r requirements.txt` <br>
4. Run the script: `python hand_tracking.py` to try out the trained model on unseen data: your hand! <br>
   Running the full game (with the AI opponent) is a little trickier. In one terminal, you must call `python server.py` and from a second terminal, you must `cd` into `web/`. From there, run `npm install` INSTALL VUE and `npm run dev`. Your current terminal will, after a little time, give some output in the form of a link which if visited, will allow you to play the game.
5. Deactivate the virtual environment once you're done: `deactivate`

**NOTE**: Make sure your virtual environment (e.g. `venv311`) is included in your `.gitignore` if you are going to push to the repository. As default, `venv311` is already included in `.gitignore`.

Note: any command containing `python` or `pip` may instead need to be executed with `python3.11` or `pip3.11`
