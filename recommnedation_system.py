import pandas as pd
import numpy as np
from matrix_decomp import MatrixDecomp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import time

class RecommenderSystem:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.train_matrix = None
        self.test_matrix = None
        self.model = None
        
    def load_and_split_data(self):
        # Load the data
        self.data = pd.read_csv(self.file_path)
        
        # Create train/test split
        train_data, test_data = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Convert to matrices
        self.train_matrix = csr_matrix(
            (train_data['rating'], 
             (train_data['user_id'], train_data['book_id']))
        ).toarray()
        
        self.test_matrix = csr_matrix(
            (test_data['rating'], 
             (test_data['user_id'], test_data['book_id']))
        ).toarray()
        
        return self.train_matrix, self.test_matrix
    
    def train_model(self, n_factors=20, reg=0.1, learning_rate=0.01, iterations=50):
        self.model = MatrixDecomp(
            ratings=self.train_matrix,
            n_factors=n_factors,
            reg=reg,
            learning_rate=learning_rate,
            iterations=iterations
        )
        self.model.train()
        return self.model
    
    def predict_rating(self, user_id, book_id):
        """Predict rating for a specific user-book pair"""
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
        return self.model.predict(user_id, book_id)
    
    def get_top_n_recommendations(self, user_id, n=5):
        """Get top N book recommendations for a user"""
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
            
        # Get predictions for all books for this user
        predictions = []
        for book_id in range(self.train_matrix.shape[1]):
            # Only predict for books the user hasn't rated
            if self.train_matrix[user_id, book_id] == 0:
                pred_rating = self.predict_rating(user_id, book_id)
                predictions.append((book_id, pred_rating))
        
        # Sort predictions by rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def evaluate_model(self):
        """Evaluate model performance on test set"""
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
            
        test_errors = []
        non_zero_positions = np.nonzero(self.test_matrix)
        
        for user, book in zip(non_zero_positions[0], non_zero_positions[1]):
            actual = self.test_matrix[user, book]
            predicted = self.predict_rating(user, book)
            test_errors.append((actual - predicted) ** 2)
            
        mse = np.mean(test_errors)
        rmse = np.sqrt(mse)
        return {'MSE': mse, 'RMSE': rmse}

# Example usage
# def main():
#     # Initialize recommender
#     recommender = RecommenderSystem(file_path="ratings.csv")
    
#     # Load and split data
#     train_matrix, test_matrix = recommender.load_and_split_data()
#     print(f"Train matrix shape: {train_matrix.shape}")
#     print(f"Test matrix shape: {test_matrix.shape}")
    
#     # Train model
#     model = recommender.train_model(
#         n_factors=10,
#         reg=0.1,
#         learning_rate=0.01,
#         iterations=20
#     )
    
#     # Get recommendations for a specific user
#     user_id = 8  # Example user
#     recommendations = recommender.get_top_n_recommendations(user_id, n=5)
#     print(f"\nTop 5 recommendations for user {user_id}:")
#     for book_id, predicted_rating in recommendations:
#         print(f"Book ID: {book_id}, Predicted Rating: {predicted_rating:.2f}")
    
#     # Evaluate model
#     metrics = recommender.evaluate_model()
#     print("\nModel Performance:")
#     print(f"MSE: {metrics['MSE']:.4f}")
#     print(f"RMSE: {metrics['RMSE']:.4f}")

# if __name__ == "__main__":
#     main()