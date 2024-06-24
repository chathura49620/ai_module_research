from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import os
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import javalang as jl

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def __get_start_end_for_node(node_to_find , tree):
    start = None
    end = None
    for path, node in tree:
        if start is not None and node_to_find not in path:
            end = node.position
            return start, end
        if start is None and node == node_to_find:
            start = node.position
    return start, end


def __get_string(start, end , data):
    if start is None:
        return ""

    # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
    end_pos = None
    start_pos = None

    if end is not None:
        end_pos = end.line - 1

    if start is not None:
        start_pos = start.line - 1

        

    lines = data.splitlines(True)
    string = "".join(lines[start.line:end_pos])
    string = lines[start.line - 1] + string

    # When the method is the last one, it will contain a additional brace
    if end is None:
        left = string.count("{")
        right = string.count("}")
        if right - left == 1:
            p = string.rfind("}")
            string = string[:p]

    return string

class Item(BaseModel):
    directoryPath: str

class FilePathForMetrics(BaseModel):
    filePath: str


class DataPredectDabites(BaseModel):
    height: int

class DataPredectLungCancer(BaseModel):
    height: int

class DataPredectHeartAttack(BaseModel):
    height: int





@app.post("/predect-daibiteis/")
async def getPredectDabites(datPredectDabites: DataPredectDabites):
    with open("./trained_models/diabites.txt", "rb") as file:
        data = pickle.load(file)
        regressor_loaded = data["model"]  
        is_daibites = regressor_loaded.predict(np.array([
            [
             
            ]
        ]))

        return {"is_have_daibites"  : is_daibites.tolist()}
    

@app.post("/predect-lung-cancer/")
async def getPredectLungCancer(DataPredectLungCancer: DataPredectLungCancer):
    with open("./trained_models/lung_cancer.txt", "rb") as file:
        data = pickle.load(file)
        regressor_loaded = data["model"]  
        is_lung_cancer = regressor_loaded.predict(np.array([
            [
             
            ]
        ]))

        return {"is_have_lung_cancer"  : is_lung_cancer.tolist()}
    
@app.post("/predect-heart-attack/")
async def getPredectHeartAttack(DataPredectHeartAttack: DataPredectHeartAttack):
    with open("./trained_models/heart_attack.txt", "rb") as file:
        data = pickle.load(file)
        regressor_loaded = data["model"]  
        is_heart_attack = regressor_loaded.predict(np.array([
            [
             
            ]
        ]))

        return {"is_have_heart_attack"  : is_heart_attack.tolist()}
