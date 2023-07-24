#bring in lightweight dependencies

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas

app=FastAPI()

class ScoringItem(BaseModel):
    YearsAtCompany:float
    EmployeeSatisfaction:float
    Position:str
    Salary:int
    
with open('rfmodel.pkl','rb') as f:
    model=pickle.load(f)    

   
@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pandas.DataFrame([item.dict()])  # This creates a DataFrame with a single row and four columns
    yhat = model.predict(df)
    return {"prediction": int(yhat[0])}
   
