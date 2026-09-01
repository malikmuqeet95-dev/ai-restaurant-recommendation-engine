# 1. Data & Schema Layer: Create a structured menu dataset with allergen, diet, price, and category metadata.
# 2. Core LLM Engine: Implement OpenAI structured outputs (Pydantic / JSON schema) to generate context-aware pairings, upsells, and dietary-safe items.
# 3. Defensive Validation: Implement input sanitation (empty/malformed queries) and out-of-catalog hallucination guards.
# 4. Evaluation & Delivery: Benchmark across 10 distinct test cases (allergens, budget, edge queries), document metrics in README, and record screen walkthrough.


import os
import json
import sys
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from config import GROQ_API_KEY

# -------------------------------------------------------------
# 1. API Client Initialization
# -------------------------------------------------------------
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in config.py or environment variables.")

# 1. Initialize Client pointing to Groq's free endpoint
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Prefer an explicit model from the environment, otherwise use models that
# currently support Groq Structured Outputs.
MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL"),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
MODEL_CANDIDATES = [model for model in MODEL_CANDIDATES if model]

# -------------------------------------------------------------
# 2. Pydantic Output Schema (Ensures deterministic JSON output)
# -------------------------------------------------------------
class RecommendedItem(BaseModel):
    item_id: str = Field(description="Unique ID of the menu item (e.g., M1, S2)")
    item_name: str = Field(description="Exact name from the catalog")
    category: str = Field(description="Mains, Sides, Beverages, or Dessert")
    price: float = Field(description="Price in USD")
    pairing_reason: str = Field(description="Appetizing rationale matching user constraints")

class RecommendationResponse(BaseModel):
    user_intent_summary: str = Field(description="Short summary of what user requested")
    dietary_flags_respected: List[str] = Field(description="List of allergies/diets filtered")
    total_estimated_price: float = Field(description="Total cost of recommended items")
    recommendations: List[RecommendedItem] = Field(description="List of selected items")
    upsell_suggestion: Optional[str] = Field(None, description="Contextual pairing upsell (drink/dessert)")

# -------------------------------------------------------------
# 3. Sample Restaurant Menu Catalog
# -------------------------------------------------------------
RESTAURANT_MENU = [
    {
        "id": "M1",
        "name": "Truffle Mushroom Burger",
        "category": "Mains",
        "price": 14.50,
        "diet": ["Vegetarian"],
        "allergens": ["Gluten", "Dairy"],
        "description": "Brioche bun, sautéed wild mushrooms, aged swiss, truffle aioli."
    },
    {
        "id": "M2",
        "name": "Smoked BBQ Brisket Platter",
        "category": "Mains",
        "price": 19.00,
        "diet": ["High-Protein", "Halal"],
        "allergens": [],
        "description": "12-hour oak-smoked beef brisket, house slaw, pickles."
    },
    {
        "id": "M3",
        "name": "Avocado Quinoa Power Bowl",
        "category": "Mains",
        "price": 12.50,
        "diet": ["Vegan", "Gluten-Free"],
        "allergens": [],
        "description": "Organic quinoa, roasted chickpeas, Hass avocado, lemon-tahini."
    },
    {
        "id": "S1",
        "name": "Crispy Sweet Potato Fries",
        "category": "Sides",
        "price": 5.50,
        "diet": ["Vegan", "Gluten-Free"],
        "allergens": [],
        "description": "Hand-cut sweet potatoes dusted with smoked paprika."
    },
    {
        "id": "S2",
        "name": "Garlic Parmesan Truffle Fries",
        "category": "Sides",
        "price": 6.50,
        "diet": ["Vegetarian"],
        "allergens": ["Dairy"],
        "description": "Crispy russet fries tossed in roasted garlic, parmesan, truffle oil."
    },
    {
        "id": "D1",
        "name": "Hibiscus Lemonade Sparkler",
        "category": "Beverages",
        "price": 4.50,
        "diet": ["Vegan", "Gluten-Free"],
        "allergens": [],
        "description": "Cold-brewed hibiscus tea with fresh sparkling lemonade."
    },
    {
        "id": "D2",
        "name": "Iced Oat Milk Matcha Latte",
        "category": "Beverages",
        "price": 5.00,
        "diet": ["Vegan", "Gluten-Free"],
        "allergens": [],
        "description": "Uji ceremonial matcha with barista-blend oat milk."
    },
    {
        "id": "DS1",
        "name": "Warm Molten Lava Cake",
        "category": "Dessert",
        "price": 7.50,
        "diet": ["Vegetarian"],
        "allergens": ["Dairy", "Gluten", "Eggs"],
        "description": "Dark Belgian chocolate cake with vanilla bean gelato."
    }
]

# -------------------------------------------------------------
# 4. Core Recommendation Function
# -------------------------------------------------------------
def get_recommendations(user_query: str, user_history: Optional[List[str]] = None) -> RecommendationResponse:
    if not user_query or not isinstance(user_query, str):
        raise ValueError("Invalid input: Query cannot be empty.")
    
    cleaned_query = user_query.strip()
    if len(cleaned_query) < 3:
        raise ValueError("Invalid input: Query is too short to generate a recommendation.")

    system_prompt = f"""
    You are an intelligent restaurant recommendation engine for a modern dining chain.
    Your job is to recommend items strictly from the MENU CATALOG below.

    MENU CATALOG:
    {json.dumps(RESTAURANT_MENU, indent=2)}

    OPERATIONAL RULES:
    1. NEVER suggest items that do not exist in the MENU CATALOG.
    2. STRICT ALLERGEN SAFETY: If the user indicates an allergy or diet (e.g., Dairy, Gluten, Vegan), NEVER include items containing those allergens.
    3. BUDGET AWARENESS: Stay within the user's budget if mentioned.
    4. UPSELLING: Suggest complementary pairings (like pairing a side or drink with a main).
    5. Output strictly according to the specified JSON schema.
    """

    user_payload = {
        "user_query": cleaned_query,
        "order_history": user_history or []
    }

    try:
        last_error: Optional[Exception] = None

        for model_name in MODEL_CANDIDATES:
            try:
                completion = client.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload)}
                    ],
                    response_format=RecommendationResponse,
                    temperature=0.2
                )
                return completion.choices[0].message.parsed
            except Exception as model_error:
                last_error = model_error
                # Try the next candidate when a configured model is unavailable.
                error_text = str(model_error).lower()
                unavailable_model = any(
                    marker in error_text
                    for marker in (
                        "model not found",
                        "model_not_found",
                        "model decommissioned",
                        "model_decommissioned",
                    )
                )
                if not unavailable_model:
                    raise RuntimeError(f"OpenAI API Execution Error: {model_error}") from model_error

        raise RuntimeError(
            f"OpenAI API Execution Error: unable to use any candidate model "
            f"({', '.join(MODEL_CANDIDATES)}). Last error: {last_error}"
        )

    except ValidationError as ve:
        raise RuntimeError(f"Output Schema Violation: {ve}")

# -------------------------------------------------------------
# 5. Execution Block
# -------------------------------------------------------------
if __name__ == "__main__":
    print(">>> Starting Recommendation Engine Demo...", flush=True)
    
    demo_queries = [
        "I am strictly vegan and need lunch with a refreshing drink under $20.",
        "Looking for a high protein dinner with no dietary restrictions, budget around $25."
    ]

    for idx, query in enumerate(demo_queries, start=1):
        print("\n" + "=" * 60, flush=True)
        print(f"QUERY {idx}: {query}", flush=True)
        print("=" * 60, flush=True)
        try:
            print("Sending request to OpenAI API...", flush=True)
            output = get_recommendations(query)
            print("Response received successfully:\n", flush=True)
            print(json.dumps(output.model_dump(), indent=2), flush=True)
        except Exception as err:
            print(f"ERROR: {err}", file=sys.stderr, flush=True)

    print("\n>>> Execution completed.", flush=True)
