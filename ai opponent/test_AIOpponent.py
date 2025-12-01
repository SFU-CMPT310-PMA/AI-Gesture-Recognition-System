import numpy as np
import unittest
from ai_opponent import AIOpponent

"""
    A unit test to check if the class AIOpponent works and return expected type
    COMMAND LINE TO RUN: python -m unittest test_AIOpponent.py
"""

class TestAIOpponent(unittest.TestCase):
    def test_initialization(self):
        ai = AIOpponent()
        self.assertEqual(ai.states, ["Rock", "Paper", "Scissors"], "State Initialization is NOT Paper, Rock, and Scissors")
        self.assertTrue(np.array_equal(ai.counts, np.ones((3,3))), "Count Initialization is NOT a matrix 3x3 of 1")
        self.assertEqual(ai.user_last_gesture, None, "Last Move Initialization is NOT None")
        self.assertEqual(ai.against_move, {'Rock' : 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}, "Against Move Dictionary is WRONG")

    def test_updateLastGesture(self):
        ai = AIOpponent()
        ai.updateLastGesture(1) #Update to Paper
        self.assertEqual(ai.user_last_gesture, 1)

    def test_updateCounts(self):
        ai = AIOpponent()
        ai.updateLastGesture(1) # Prev gesture = Paper
        ai.updateCounts(2) # Current gesture = Scissors so they played Paper -> Scissors
        self.assertEqual(ai.counts[1][2], 2)
        

    def test_prediction(self):
        #Gesture Sequence User plays: Paper -> Scissors -> Paper
        ai = AIOpponent()
        ai.updateLastGesture(1) # Prev gesture = Paper
        ai.updateCounts(2) # Current gesture = Scissors

        pred = ai.predictNextGesture()
        self.assertIn(pred, ai.states, "Not in the available states")
        

    def test_AInextGesture(self):
        ai = AIOpponent()
        user_gestures = [0, 1, 1, 2, 1, 0, 0, 2, 1, 2, 0, 1, 2, 0, 0, 2]
        correctness = 0
        i = 0
        possible_move = ai.states
        for move in user_gestures:
            print("Round ", i)

            if ai.user_last_gesture is None:
                ai_move = np.random.choice(possible_move)
                print("User: ", possible_move[move], ", AI: ", ai_move)
            else:
                user_pred = ai.predictNextGesture()
                ai_move = ai.updateAINextGesture(user_pred)
                print("User Prediction: ", user_pred)
                print()
                print("******REAL RESULT:******")
                print("User: ", possible_move[move], ", AI: ", ai_move)
                
                if (user_pred == possible_move[move]):
                    correctness+= 1

            #Update last round result to the system
            i+= 1
            if ai.user_last_gesture is not None:
                ai.updateCounts(move)
            ai.updateLastGesture(move)
            print("------------------------------------")
        
        print("The prediction accuracy is: ", (correctness/(len(user_gestures) - 1)) * 100, "%")

if __name__ == '__main__':
    unittest.main()