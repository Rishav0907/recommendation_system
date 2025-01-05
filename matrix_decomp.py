import time
import numpy as np
from matplotlib import pyplot as plt

class MatrixDecomp:
    def __init__(self, ratings, n_factors, reg, learning_rate, iterations):
        self.ratings = ratings
        self.num_users = ratings.shape[0]
        self.num_items = ratings.shape[1]
        self.total_user_item_interaction = np.count_nonzero(ratings)  # Fixed variable name
        self.nonzero_interac_list = list(range(self.total_user_item_interaction))
        self.non_zero_row, self.non_zero_col = ratings.nonzero()
        self.latent_features = n_factors
        self.reg = reg
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.mse_epoch = []
        self.stop = False
        
    def initialize(self):
        self.now = time.time()
        # Initialize with smaller random values
        self.user_bias = np.random.normal(0, 0.1, size=self.num_users)
        self.item_bias = np.random.normal(0, 0.1, size=self.num_items)
        self.user_decomposed_matrix = np.random.normal(0, 0.1/np.sqrt(self.latent_features), 
                                                      size=(self.num_users, self.latent_features))
        self.item_decomposed_matrix = np.random.normal(0, 0.1/np.sqrt(self.latent_features), 
                                                      size=(self.num_items, self.latent_features))

    def update_parameters(self, error, user, item):
        # print("Updating Params")
        # Add clipping to prevent extreme values
        clip_value = 5.0
        
        # Update biases with clipping
        user_bias_grad = error - (self.reg * self.user_bias[user])
        item_bias_grad = error - (self.reg * self.item_bias[item])
        
        self.user_bias[user] += self.learning_rate * np.clip(user_bias_grad, -clip_value, clip_value)
        self.item_bias[item] += self.learning_rate * np.clip(item_bias_grad, -clip_value, clip_value)
        
        # Update matrices with clipping
        user_matrix_grad = error * self.item_decomposed_matrix[item] - self.reg * self.user_decomposed_matrix[user]
        item_matrix_grad = error * self.user_decomposed_matrix[user] - self.reg * self.item_decomposed_matrix[item]
        
        self.user_decomposed_matrix[user] += self.learning_rate * np.clip(user_matrix_grad, -clip_value, clip_value)
        self.item_decomposed_matrix[item] += self.learning_rate * np.clip(item_matrix_grad, -clip_value, clip_value)

    def predict(self, user, item):
        # print("prediciting")
        prediction = (self.user_bias[user] + 
                     self.item_bias[item] + 
                     np.dot(self.user_decomposed_matrix[user], self.item_decomposed_matrix[item]))
        # Clip predictions to prevent extreme values
        return np.clip(prediction, self.ratings.min(), self.ratings.max())

    def evaluate(self, epoch):
        # print("Evaluating")
        total_error = 0.0
        for i in self.nonzero_interac_list[:10]:
            user = self.non_zero_row[i]
            item = self.non_zero_col[i]
            prediction = self.predict(user, item)
            sq_error = (prediction - self.ratings[user, item]) ** 2
            total_error += sq_error
            
        if self.total_user_item_interaction > 0:  # Ensure we don't divide by zero
            mse = total_error / self.total_user_item_interaction
            self.mse_epoch.append(float(mse))  # Ensure we're storing a float
            print(f"---> Epoch {epoch}")
            temp = np.round(time.time() - self.now, 3)
            print(f"ave mse {np.round(self.mse_epoch[-1], 3)} ===> Total training time: {temp} seconds.")
        else:
            print("Error: No non-zero interactions found in the ratings matrix")

    def train(self):
        self.initialize()
        for epoch in range(1, self.iterations):
            # print(f"Epoch {epoch} running")
            np.random.shuffle(self.nonzero_interac_list)
            if not self.stop:
                for index in self.nonzero_interac_list[:10]:
                    print(index)
                    user, item = self.non_zero_row[index], self.non_zero_col[index]
                    pred_rat = self.predict(user, item)
                    error = self.ratings[user, item] - pred_rat
                    self.update_parameters(error, user, item)
                self.evaluate(epoch)
        self.plot_the_score()
    
    def plot_the_score(self):
        plt.figure(figsize=(18, 6))
        plt.plot(range(1, 1 + len(self.mse_epoch)), self.mse_epoch, marker='o')
        plt.title("SGD Custom Prepared USER & ITEM vector's Tr MSE loss vs epochs", fontsize=20)
        plt.xlabel('Number of epochs', fontsize=18)
        plt.ylabel('Mean Square Error', fontsize=18)
        plt.xticks(range(1, len(self.mse_epoch) + 2), fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.show()