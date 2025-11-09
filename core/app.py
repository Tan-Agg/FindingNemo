# import streamlit as st
# import json
# import os
# from router  import IntentRouter   # your class file

# LOG_PATH = "data/query_log.json"

# # -----------------------------
# # 1. Initialize router + setup
# # -----------------------------
# router = IntentRouter()

# # Ensure data directory exists
# os.makedirs("data", exist_ok=True)

# # Load or create log file
# if not os.path.exists(LOG_PATH):
#     with open(LOG_PATH, "w") as f:
#         json.dump([], f)

# # -----------------------------
# # 2. Streamlit UI
# # -----------------------------
# st.title("Intent Label Generator")
# st.write("Enter a user query below to generate and log its intent label and description.")

# user_input = st.text_area("Enter a query:", placeholder="e.g., Write me a Python script to extract keywords")

# if st.button("Generate Intent"):
#     if user_input.strip():
#         with st.spinner("Analyzing and generating intent..."):
#             intent_data = router.generate_intent(user_input)

#         st.success("Intent Generated!")
#         st.json(intent_data)

#         # -----------------------------
#         # 3. Append to query_log
#         # -----------------------------
#         with open(LOG_PATH, "r") as f:
#             data = json.load(f)

#         data.append({
#             "user_query": user_input,
#             "intent_label": intent_data["intent_label"],
#             "description": intent_data["description"],
#             "confidence": intent_data.get("confidence", 0.0)
#         })

#         with open(LOG_PATH, "w") as f:
#             json.dump(data, f, indent=2)

#         st.info("Logged to `data/query_log.json`")

#     else:
#         st.warning("Please enter a valid query first.")


import streamlit as st
from embeddings import EmbeddingService
from memory_bank import MemoryBank
import sys
sys.path.append('..')
from config import SIMILARITY_THRESHOLD

# Initialize services
@st.cache_resource
def load_services():
    embedding_service = EmbeddingService()
    memory_bank = MemoryBank()
    return embedding_service, memory_bank

embedding_service, memory_bank = load_services()

# Streamlit UI
st.title("🎯 Agent Matcher")
st.write("Enter a query to check if a specialist agent exists")

# Show available specialists
with st.expander("📊 Available Specialists"):
    specialists = memory_bank.get_all_specialists()
    if specialists:
        for spec in specialists:
            st.write(f"**{spec['intent_label']}**: {spec['description']}")
    else:
        st.write("No specialists available")

# Input
user_query = st.text_area("Enter your query:", placeholder="e.g., Plan a trip to Tokyo")

if st.button("Find Agent"):
    if user_query.strip():
        with st.spinner("Searching for matching agent..."):
            # Create embedding
            query_embedding = embedding_service.create_embedding(user_query)
            
            # Search memory bank
            result = memory_bank.search(query_embedding)
        
        if result:
            st.success("✅ Agent Found!")
            st.write(f"**Agent**: {result['specialist']['intent_label']}")
            st.write(f"**Description**: {result['specialist']['description']}")
            st.write(f"**Similarity**: {result['similarity']:.4f}")
            st.write(f"**Threshold**: {SIMILARITY_THRESHOLD}")
            st.write(f"**Endpoint**: {result['specialist']['endpoint']}")
            
            # Show metadata if exists
            if result['specialist'].get('metadata'):
                st.json(result['specialist']['metadata'])
        else:
            st.warning("❌ No matching agent found")
            st.write(f"Threshold: {SIMILARITY_THRESHOLD}")
            st.info("💡 Try a different query or add a new specialist")
    else:
        st.warning("Please enter a query")