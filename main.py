import fastapi
from recommnedation_system import RecommenderSystem
from pydantic import BaseModel
from typing import List,Dict

app=fastapi.FastAPI(title="Book Recommender API")

FILE_PATH="ratings.csv"
recommender=RecommenderSystem(FILE_PATH)

train_matrix, test_matrix = recommender.load_and_split_data()
    
# Train model
recommender.train_model(
    n_factors=10,
    reg=0.1,
    learning_rate=0.01,
    iterations=20
)

class Rating_req(BaseModel):
    user_id: int
    book_id: int

class Recommendation_response(BaseModel):
    book_id: int
    predicted_rating: float

@app.get("/")
async def root():
    return {"message":"Book recommendation system"}

@app.get("/predict",response_model=float)
async def predict_rating(request: Rating_req):
    try:
        pred=recommender.predict()
        return float(pred)
    except Exception as e:
        raise fastapi.HTTPException(status_code=400,detail=str(e))
    

@app.get("/recommend/{user_id}", response_model=List[Recommendation_response])
async def get_recommendations(user_id: int, n: int = 5):
    try:
        recommendations = recommender.get_top_n_recommendations(user_id, n)
        return [
            Recommendation_response(book_id=book_id, predicted_rating=float(rating))
            for book_id, rating in recommendations
        ]
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    
@app.get("/evaluate", response_model=Dict[str, float])
async def evaluate():
    try:
        return recommender.evaluate_model()
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))