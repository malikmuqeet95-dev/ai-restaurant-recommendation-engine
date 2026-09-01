# AI Product Recommendation Engine for a Restaurant Chain

Component: Core AI Recommendation Subsystem, Dynamic Prompting & Structured Schema Parsing  
Primary Stack: Python 3.10+, Groq API (OpenAI Compatible SDK), Pydantic v2, Python-Dotenv  

---

1. Project Overview

This deliverable contains the core recommendation engine backend for a restaurant chain. Rather than generating unconstrained conversational text, the engine processes natural language user queries (dietary restrictions, strict allergies, budget limits, meal types) and returns deterministically validated JSON objects strictly bound to the restaurant's active menu catalog.

---

2. Technical Architecture & Design Decisions

- In-Context Catalog Serialization: Directly serializes and injects `RESTAURANT_MENU` into the system prompt to eliminate off-menu hallucinations.

- Model Fallback Routing: Implements dynamic candidate failover across `MODEL_CANDIDATES` (`os.getenv("GROQ_MODEL")`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`) to recover gracefully if a specific model endpoint is unavailable or decommissioned.

- Strict Schema Enforcement: Uses Pydantic (`RecommendationResponse` and `RecommendedItem`) with native structured parsing (`client.chat.completions.parse`) to guarantee field types, numeric prices, and pairing rationale.

- Pre-API Defensive Validation: Inspects and sanitizes user input prior to making remote API calls to prevent blank or trivial requests from consuming token quota.

- Deterministic Business Logic: Enforces zero-tolerance allergen filtering, budget threshold tracking, and multi-course meal upselling directly in the prompt layer.

---

**Setup & Installation**

1. Clone the Repository

2. Create and Activate Virtual Environment  
 python -m venv .venv
 .venv\Scripts\activate

3. Install Dependencies  
 pip install -r requirements.txt

4. Configure Environment Variables
 Create a .env file in project root

5. Run the Application
 python recommendation_engine.py

//Run the 10-case evaluation test suite
 python test_engine.py
