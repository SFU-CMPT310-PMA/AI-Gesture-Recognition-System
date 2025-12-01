import numpy as np

class AIOpponent:
    """
        Predict and Move based on the last gesture/ move from the user
        Adapt from: https://www.geeksforgeeks.org/machine-learning/markov-chain/
    """

    def __init__(self):
        self.states = ["Rock", "Paper", "Scissors"]
        self.counts = np.ones((3,3))
        self.user_last_gesture = None  #int - Can be the index from [0, 1, 2] corresponding to index of states
        self.against_move = {'Rock' : 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
    
    def updateLastGesture(self, last_gesture):
        """
        Argument: self = AIOpponent, last_gesture = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: Update the last move to the current (Previous Gesture)
        """
        self.user_last_gesture = last_gesture
    

    def updateCounts(self, current_gesture):
        """
        Argument: self = AIOpponent, last_gesture = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: Update new counts of user gesture to counts (Current Gesture)
        """
        #Add new counts to the table (matrix)
        self.counts[self.user_last_gesture][current_gesture] += 1    

    def predictNextGesture(self):
        """
        Argument: self = AIOpponent, last_gesture = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: 
            1. Get prior probability with Dirichlet Distribution
            2. Predict user's next gesture based on the history of user's move
            3. Return the predicted user state in the next step

        Note: [0, 1, 2] ([Rock, Paper, Scissor]) = index corresponding to states variable
        """
        probability = np.random.dirichlet(self.counts[self.user_last_gesture])
        next_state = np.random.choice([0, 1, 2], p=probability)
        predicted_user_gesture = self.states[next_state]
        return predicted_user_gesture
    

    def updateAINextGesture(self, predicted_user_gesture):
        """
        Argument: self = AIOpponent, predicted_user_gesture = "Rock", "Paper" or "Scissors"
        Goal: Return the move to play against human
        """
        return self.against_move[predicted_user_gesture]


########################################    BAYESIAN BASED ON LAST 2 STEPS  ########################################
class BayesianAIOpponent():
    """
        Predict and Move based on the last 2 gestures/ moves from the user
    """

    def __init__(self):
        self.states = ["Rock", "Paper", "Scissors"]
        self.counts = np.ones((3, 3, 3))
        self.user_last_two = []  #int array with value of index from [0, 1, 2] corresponding to index of states
        self.against_move = {'Rock' : 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
    
    def update(self, last_gesture):
        """
        Argument: self = AIOpponent, last_gesture = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: Update the last 2 moves to the current (2 Previous Gesture)
        """
        #Keep track last 2 moves with the current move
        if len(self.user_last_two) >= 2:
            self.user_last_two.pop(0) #only keep last 2
        self.user_last_two.append(last_gesture)

        #Update the counts
        if len(self.user_last_two) == 2:
            prev2, prev1 = self.user_last_two
            self.counts[prev2][prev1][last_gesture] += 1

    def predictNextGesture(self):
        """
        Argument: self = AIOpponent, user_move = 0, 1 or 2 corresponding to "Rock", "Paper" or "Scissors"
        Goal: Return a string of "Rock", "Paper" or "Scissors" for User Next Gesture Prediction
        """
        # Random the first 2 round
        if len(self.user_last_two) < 2:
            next_state = np.random.choice([0, 1, 2])
            return self.states[next_state]
        
        #Get the last 2 rounds gesture
        prev2, prev1 = self.user_last_two[0], self.user_last_two[1]

        # Predict
        probability = np.random.dirichlet(self.counts[prev2][prev1])
        next_state = np.random.choice([0, 1, 2], p=probability)
        predicted_user_gesture = self.states[next_state]
        return predicted_user_gesture

    def updateAINextGesture(self, predicted_user_gesture):
        """
        Argument: self = AIOpponent, predicted_user_gesture = "Rock", "Paper" or "Scissors"
        Goal: Return a string of the move to play against human
        """
        return self.against_move[predicted_user_gesture]

        


    
   