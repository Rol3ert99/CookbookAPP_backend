from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

cuisines = ["African", "American", "Asian", "British", "Caribbean", "Chinese", 
            "Eastern European", "French", "German", "Greek", "Indian", 
            "Italian", "Japanese", "Korean", "Latin American", "Mediterranean", 
            "Mexican", "Middle Eastern", "Nordic", "Polish", "Portuguese", 
            "Russian", "Spanish", "Thai", "Vietnamese", "Other"]

categories = ["Appetizer", "Main Course", "Dessert", "Soup", "Salad", 
              "Side Dish", "Snack", "Beverage", "Breakfast", "Brunch", "Lunch", 
              "Dinner", "Vegetarian", "Vegan", "Gluten Free", "Dairy Free", 
              "Holiday", "Christmas", "Easter", "Halloween", "Valentine's Day", 
              "Summer", "Winter", "Spring", "Autumn", "Seafood", "Meat", 
              "Poultry", "Pasta", "Rice", "Stew", "Bake", "Other"]

# Ensure you have set the OPENAI_API_KEY in your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    temperature=0, 
    model_name="gpt-4o-mini"
)


class IdeasRequest(BaseModel):
    ingredients: list

class StepsRequest(BaseModel):
    name: str
    ingredients: list


@app.post("/ideas")
def get_ideas(payload: IdeasRequest):

    ideas_prompt = PromptTemplate.from_template(
        """
        You are an AI consultant specializing in kitchen ideas
        Based on the following list of ingredients, create dish suggestions. Each dish should meet the following requirements:

        1. The dishes can only use ingredients from the provided list.
        2. Each dish should be assigned to one of the categories from the provided list.
        3. Determine the cuisine for each dish based on the provided list of cuisines.
        4. Provide an estimated preparation time for each dish.
        5. For each dish, give a short description including key details such as taste, texture.
        6. Return the result in JSON format according to the structure below.

        ### List of ingredients:
        {ingredients}

        ### Dish categories:
        {categories}

        ### Types of cuisine:
        {cuisines}

        ### Response structure:
        {{
        "dishes": [
            {{
            "name": "Dish name",
            "category": "Category from the provided list",
            "cuisine": "Cuisine type from the provided list",
            "time": "Preparation time in minutes",
            "description": "Dish description"
            }},
            ...
        ]
        }}
        Your response must be a valid JSON object in the above format, and nothing else.
        Make sure you select only one category from the available list.
        Make sure you only select one cuisine type from the list provided.
        Ensure that we have all the ingredients available for the dish.
        IMPORTANT: Do not include any explanations or additional text outside of this JSON object.
    """)

    ideas_chain = ideas_prompt | llm

    ingredients = payload.ingredients

    result = ideas_chain.invoke({
        "ingredients": ingredients,
        "categories": categories,
        "cuisines": cuisines
    })

    ideas = json.loads(result.content)
    return ideas


@app.post("/steps")
def get_steps(payload: StepsRequest):
    steps_prompt = PromptTemplate.from_template(
        """
        You are an AI consultant specializing in cooking.
        Prepare the steps for preparing the dish. Below you have the name of the dish and all the necessary ingredients.
        Return the result in JSON format according to the structure below.

        dish name: {dish_name}

        ingredients: {ingredients}

        Response structure:
        {{
        "steps": [{{1: "step 1 text"}}, {{2: "step 2 text"}}, ...]
        }}

        Your response must be a valid JSON object in the above format, and nothing else.
        IMPORTANT: Do not include any explanations or additional text outside of this JSON object.
        """
    )

    dish_name = payload.name
    ingredients = payload.ingredients

    steps_chain = steps_prompt | llm

    result = steps_chain.invoke({
        "dish_name": dish_name,
        "ingredients": ingredients
    })

    steps = json.loads(result.content)
    return steps