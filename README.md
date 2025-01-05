# AI-Powered Book Recommendation Engine: FastAPI & Docker

A scalable recommendation system that provides personalized book suggestions using collaborative filtering and matrix factorization, deployed as a REST API and containerized with Docker.

## Features

- Matrix decomposition-based collaborative filtering
- Real-time rating predictions and recommendations
- RESTful API endpoints using FastAPI
- Docker containerization for easy deployment
- Model performance evaluation metrics
- Automatic train/test data splitting
- Configurable model hyperparameters

## Tech Stack

- Python 3.9
- FastAPI
- NumPy
- Pandas
- SciPy
- scikit-learn
- Docker
- uvicorn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd book-recommender
```

2. Using Docker:
```bash
docker build -t book-recommender .
docker run -p 8000:8000 book-recommender
```

3. Manual installation:
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Root
- `GET /`: Health check endpoint

### Predict Rating
- `GET /predict`: Predicts rating for a user-book pair
- Request body:
```json
{
    "user_id": 123,
    "book_id": 456
}
```

### Get Recommendations
- `GET /recommend/{user_id}`: Gets top N book recommendations for a user
- Query parameters:
  - `n`: Number of recommendations (default: 5)

### Evaluate Model
- `GET /evaluate`: Returns model performance metrics (MSE and RMSE)

## Model Details

The recommendation system uses matrix factorization to decompose the user-item interaction matrix into lower-dimensional user and item matrices. The model:

- Learns latent features for users and books
- Incorporates user and item biases
- Uses stochastic gradient descent for optimization
- Includes regularization to prevent overfitting
- Implements dynamic learning rate adjustment

## Configuration

Key model parameters can be configured when training:

- `n_factors`: Number of latent factors
- `reg`: Regularization strength
- `learning_rate`: Initial learning rate
- `iterations`: Number of training iterations

## Dataset

The system expects a CSV file with the following columns:
- `user_id`: Unique identifier for users
- `book_id`: Unique identifier for books
- `rating`: Rating value

## Performance

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
