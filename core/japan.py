from embeddings import EmbeddingService
import json

# Initialize embedding service
embedding_service = EmbeddingService()

# Load memory bank
with open('../data/memory_bank.json', 'r') as f:
    data = json.load(f)

# Remove old Japan specialist if exists
data['specialists'] = [s for s in data['specialists'] if s['intent_label'] != 'japan_travel']

# Create Japan specialist with REAL embeddings
description = "Specialist for comprehensive Japan travel and cultural guidance. Provides expert assistance with trip planning including itinerary creation, transportation (JR Pass, trains, buses), accommodation recommendations, and budget planning. Offers insights on Japanese culture, traditions, etiquette, festivals, and seasonal events. Advises on cuisine including restaurant recommendations, regional specialties, food etiquette, and dietary considerations. Covers major destinations (Tokyo, Kyoto, Osaka, Hiroshima, Hokkaido, Okinawa) and hidden gems. Provides practical information on visas, currency, language basics, SIM cards, and local customs. Assists with activity planning including temples, shrines, museums, nature spots, shopping districts, and entertainment venues."
embedding = embedding_service.create_embedding(description)

specialist = {
    "intent_label": "japan_travel",
    "description": description,
    "endpoint": "nvidia/llama-3.1-nemotron-70b-instruct",  # ← CHANGE THIS LINE
    "embedding": embedding,
    "metadata": {"region": "Asia", "language": "Japanese"}
}

data['specialists'].append(specialist)

# Save
with open('../data/memory_bank.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Japan specialist added with real embeddings!")