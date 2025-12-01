import numpy as np
import unittest
from ai_opponent import BayesianAIOpponent

"""
    A unit test to check if the class BayesianAIOpponent works and return expected type
    COMMAND LINE TO RUN: python -m unittest test_bayesianAIOpponent.py
"""

class TestBayesianAIOpponent(unittest.TestCase):
    def test_prediction(self):
        #Gesture Sequence User plays: Paper -> Scissors -> Paper
        ai = BayesianAIOpponent()
        ai.update(1) # last_gesture = 1
        ai.update(0) # last_gesture = 1
        ai.update(2) # last_gesture = 1
        
        pred = ai.predictNextGesture()
        self.assertIn(pred, ai.states, "Not in the available states")

    def test_AINextGesture(self):
        ai = BayesianAIOpponent()
        user_gestures = [0, 1, 1, 2, 1, 0, 0, 2, 1, 2, 0, 1, 2, 0, 0, 2]
        correctness = 0
        i = 0
        possible_move = ai.states

        for move in user_gestures:
            print("Round ", i)
            user_pred = ai.predictNextGesture()
            print("User Prediction: ", user_pred)
            print()

            ai_move = ai.updateAINextGesture(user_pred)
            print("******REAL RESULT:******")
            print("User: ", possible_move[move], ", AI: ", ai_move)
                
            if (user_pred == possible_move[move]):
                correctness+= 1

            #Update last round result to the system
            i+= 1
            ai.update(move)
            print("------------------------------------")
        
        print("The prediction accuracy is: ", (correctness/(len(user_gestures) - 1)) * 100, "%")
    
if __name__ == '__main__':
    unittest.main()
