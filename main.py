from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# Ensure you have set the OPENAI_API_KEY in your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

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


llm = ChatOpenAI(
    temperature=0, 
    model_name="gpt-4o-mini"
)


class IdeasRequest(BaseModel):
    ingredients: list

class StepsRequest(BaseModel):
    name: str
    ingredients: list


class SkillsRequest(BaseModel):
    skills: list
    sector: str


class User(BaseModel):
    username: str
    password: str





@app.post("/ideas")
def get_ideas(payload: IdeasRequest):

    ideas_prompt = PromptTemplate.from_template(
        """
        You are an AI consultant specializing in kitchen ideas.
        Based on the following list of ingredients, create dish suggestions. Each dish should meet the following requirements:

        1. The dishes can only use ingredients from the provided list.
        2. Each dish should be assigned to one of the categories from the provided list.
        3. Determine the cuisine for each dish based on the provided list of cuisines.
        4. Provide an estimated preparation time for each dish.
        5. For each dish, give a short description including key details such as taste and texture.
        6. Provide a list of preparation steps for each dish.
        7. Provide nutritional information for each dish, including calories, fat, protein, sugar, carbohydrates, and fiber.
        8. Provide a list of ingredients used for each dish, specifying the quantity and units, in the format:
           "ingredients": {{
               "Ingredient1": "quantity and unit",
               "Ingredient2": "quantity and unit",
               ...
           }}
        9. Provide an image description for each dish that can be used to generate an image of the finished dish.
        10. Return the result in JSON format according to the structure below.

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
                    "description": "Dish description",
                    "ingredients": {{
                        "Ingredient1": "quantity and unit",
                        "Ingredient2": "quantity and unit",
                        ...
                    }},
                    "steps": ["Step 1", "Step 2", "..."],
                    "nutrition": {{
                        "calories": "Number of calories",
                        "fat": "Amount of fat in grams",
                        "protein": "Amount of protein in grams",
                        "sugar": "Amount of sugar in grams",
                        "carbohydrates": "Amount of carbohydrates in grams",
                        "fiber": "Amount of fiber in grams"
                    }},
                    "image_description": "A detailed description of the dish's appearance for image generation"
                }},
                ...
            ]
        }}
        Your response must be a valid JSON object in the above format, and nothing else.
        Make sure you select only one category from the available list.
        Make sure you only select one cuisine type from the list provided.
        Ensure that all the ingredients used are available from the provided list.
        IMPORTANT: Do not include any explanations or additional text outside of this JSON object.
        """
    )

    ideas_chain = ideas_prompt | llm

    ingredients = payload.ingredients

    result = ideas_chain.invoke({
        "ingredients": ingredients,
        "categories": categories,
        "cuisines": cuisines
    })

    ideas = json.loads(result.content)

    # Generowanie obrazów dla każdego dania
    for dish in ideas['dishes']:
        image_description = dish['image_description']
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_description,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        image_url = response.data[0].url
        dish['image_url'] = image_url
        del dish['image_description']

    return ideas