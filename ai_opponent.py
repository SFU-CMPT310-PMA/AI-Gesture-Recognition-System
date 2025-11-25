import numpy as np

# Adapt from: https://www.geeksforgeeks.org/machine-learning/markov-chain/
class AIOpponent:
    def __init__(self):
        self.states = ["Rock", "Paper", "Scissors"]
        self.transition_step = np.ones((3,3))
        self.transition_matrix = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
        self.user_last_gesture = None  #int - Can be the index from [0, 1, 2] corresponding to index of states
        self.against_move = {'Rock' : 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
    
    def updateLastMove(self, current_gesture):
        """
        Argument: self = AIOpponent, user_move = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: Update the last move to the current
        """
        self.user_last_gesture = current_gesture
    

    def updateTransition(self, current_gesture):
        """
        Argument: self = AIOpponent, user_move = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: 
            1. Update new counts of user gesture to transition_step
            2. Update the transition_matrix (probability) based on the updated transition_step
        """
        #Add new counts to the table (matrix)
        self.transition_step[self.user_last_gesture][current_gesture] += 1

        #Update the probability
        self.transition_matrix = self.transition_step / np.sum(self.transition_step, axis = 1)
    

    def predictNextGesture(self, current_gesture):
        """
        Argument: self = AIOpponent, user_move = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: 
            1. Predict user's next gesture based on the history of user's move
            2. Return the predicted user state in the next step

        Note: [0, 1, 2] ([Rock, Paper, Scissor]) = index corresponding to states variable
        """
        next_state = np.random.choice([0, 1, 2], p=self.transition_matrix[current_gesture])
        predicted_user_gesture = self.states[next_state]
        return predicted_user_gesture
    

    def updateNextMove(self, predicted_user_gesture):
        """
        Argument: self = AIOpponent, predicted_user_gesture = "Rock", "Paper" or "Scissors"
        Goal: Return the move to play against human
        """
        return self.against_move[predicted_user_gesture]