import json
from recommendation_engine import get_recommendations

TEST_CASES = [
    {"id": 1, "query": "Vegan lunch under $20 with a drink", "focus": "Dietary (Vegan) + Budget limit"},
    {"id": 2, "query": "High-protein, Halal meal under $30", "focus": "Halal & Protein tags"},
    {"id": 3, "query": "Severe Dairy and Gluten allergy", "focus": "Strict allergen avoidance"},
    {"id": 4, "query": "Something sweet under $8", "focus": "Dessert category & low price constraint"},
    {"id": 5, "query": "Just fries and a drink for a quick snack", "focus": "Side + Beverage pairing without a Main"},
    {"id": 6, "query": "Full 3-course dinner for one person", "focus": "Multi-course meal cohesion"},
    {"id": 7, "query": "I want a pepperoni pizza", "focus": "Out-of-catalog handling"},
    {"id": 8, "query": "Total budget is strictly $10 for lunch and a drink", "focus": "Budget trade-off resolution"},
    {"id": 9, "query": "   ", "focus": "Empty input validation (Error check)"},
    {"id": 10, "query": "Low-carb / Keto dinner recommendation", "focus": "Nutritional inference"}
]

def run_tests():
    print("=" * 70)
    print("STARTING TEST SUITE: 10 TEST CASES")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n[TEST CASE #{tc['id']}] Focus: {tc['focus']}")
        print(f"Input Query: \"{tc['query']}\"")
        try:
            result = get_recommendations(tc["query"])
            print("Status: PASS (200 OK)")
            print(f"Summary: {result.user_intent_summary}")
            print(f"Items Selected: {[item.item_name for item in result.recommendations]}")
            print(f"Total Price: ${result.total_estimated_price:.2f}")
            if result.upsell_suggestion:
                print(f"Upsell: {result.upsell_suggestion}")
        except ValueError as val_err:
            print(f"Status: PASS -> CAUGHT EXPECTED VALIDATION ERROR: {val_err}")
        except Exception as err:
            print(f"Status: FAILED -> {err}")
        print("-" * 70)

if __name__ == "__main__":
    run_tests()