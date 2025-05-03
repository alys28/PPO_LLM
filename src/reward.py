

reward_values = {
    "correct": 10,
    "correct_digit": 1,
    "incorrect_digit": -1,
    "wrong_num_digits": -0.5,
}

class RewardBuilder:
    def __init__(self, reward_values=reward_values):
        self.reward_values = reward_values
    def get_reward(self, prediction, answer):
        if prediction == answer:
            return self.reward_values["correct"]
        reward = 0
        if len(prediction) != len(answer):
            # Heurisitc: penalize incrementally for each digit off
            return self.reward_values["wrong_num_digits"] * abs(len(prediction) - len(answer))
        for i in range(min(len(prediction), len(answer))):
            if prediction[i] != answer[i]:
                reward += self.reward_values["incorrect_digit"]
            else: 
                reward += self.reward_values["correct_digit"]
