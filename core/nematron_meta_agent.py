from core.router import IntentRouter
from core.embeddings import EmbeddingService
from core.memory_bank import MemoryBank
from core.model_caller import ModelCaller
from core.query_logger import QueryLogger
from core.decision_engine import DecisionEngine
from datetime import datetime
import time

class NemotronMetaAgent:
    """
    Main orchestrator - ties all components together
    This is the complete pipeline from user query to response
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("INITIALIZING NEMOTRON META-AGENT")
        print("="*60 + "\n")
        
        # Initialize all components
        print("Loading components...")
        self.router = IntentRouter()
        self.embedding_service = EmbeddingService()
        self.memory_bank = MemoryBank()
        self.model_caller = ModelCaller()
        self.query_logger = QueryLogger()
        self.decision_engine = DecisionEngine()
        
        print("\n✅ All components loaded\n")
    
    def process_query(self, user_prompt):
        """
        Main pipeline: process a user query end-to-end
        
        Args:
            user_prompt (str): User's question
            
        Returns:
            dict: Complete response with metadata
        """
        print("\n" + "="*60)
        print(f"PROCESSING QUERY: {user_prompt}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # STEP 1: Generate Intent
        print("Step 1: Generating intent...")
        intent = self.router.generate_intent(user_prompt)
        print(f"✅ Intent: {intent['intent_label']}")
        print(f"   Description: {intent['description']}\n")
        
        intent_label = intent['intent_label']
        intent_description = intent['description']
        
        # STEP 2: Create Embedding
        print("Step 2: Creating embedding...")
        query_embedding = self.embedding_service.create_embedding(intent_description)
        print(f"✅ Embedding created (384 dimensions)\n")
        
        # STEP 3: Search Memory Bank
        print("Step 3: Searching for specialist...")
        search_result = self.memory_bank.search(query_embedding)
        
        if search_result:
            # SPECIALIST FOUND
            specialist = search_result['specialist']
            similarity = search_result['similarity']
            
            print(f"✅ Specialist found: {specialist['intent_label']}")
            print(f"   Similarity: {similarity:.3f}\n")
            
            # STEP 4A: Call Specialist
            print("Step 4: Calling specialist...")
            response = self.model_caller.call_specialist(
                specialist['endpoint'],
                user_prompt
            )
            
            if response['error']:
                print(f"⚠️  Specialist failed, falling back to generalist")
                response = self.model_caller.call_generalist(user_prompt)
                routed_to = "generalist (fallback)"
            else:
                routed_to = "specialist"
                print(f"✅ Specialist responded\n")
            
            # No logging needed - specialist handled it
            training_decision = None
            
        else:
            # NO SPECIALIST FOUND
            print(f"❌ No specialist found\n")
            
            # STEP 4B: Call Generalist
            print("Step 4: Calling generalist...")
            response = self.model_caller.call_generalist(user_prompt)
            routed_to = "generalist"
            print(f"✅ Generalist responded\n")
            
            # STEP 5: Log Query
            print("Step 5: Logging query...")
            self.query_logger.log_query(intent_label, intent_description, user_prompt)
            
            # Get current count
            count = self.query_logger.get_count(intent_label)
            print(f"   Total queries for '{intent_label}': {count}\n")
            
            # STEP 6: Check if Training Needed
            print("Step 6: Checking if training needed...")
            training_decision = self.decision_engine.make_decision(intent_label, count)
            
            print(f"   Decision: {training_decision['decision']}")
            for reason in training_decision['reasons'][:3]:  # Show first 3 reasons
                print(f"   {reason}")
            print()
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        
        # Build final response
        result = {
            "answer": response['answer'],
            "metadata": {
                "intent_label": intent_label,
                "intent_description": intent_description,
                "routed_to": routed_to,
                "latency": round(latency, 3),
                "tokens_used": response.get('tokens_used', 0),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add training info if applicable
        if training_decision:
            result['metadata']['training'] = {
                "decision": training_decision['decision'],
                "count": self.query_logger.get_count(intent_label)
            }
            
            if training_decision['decision'] == "TRAIN":
                result['metadata']['training']['plan'] = self.decision_engine.get_training_plan(
                    intent_label, intent_description
                )
        
        print("="*60)
        print("QUERY PROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return result
    
    def get_system_status(self):
        """Get current system status"""
        specialists = self.memory_bank.get_all_specialists()
        logs = self.query_logger.get_all_logs()
        
        status = {
            "specialists": {
                "count": len(specialists),
                "list": [s['intent_label'] for s in specialists]
            },
            "logs": {
                "intent_types": len(logs),
                "total_queries": sum(log['count'] for log in logs.values())
            }
        }
        
        return status
    
    def print_status(self):
        """Print system status nicely"""
        status = self.get_system_status()
        
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60 + "\n")
        
        print(f"Specialists Deployed: {status['specialists']['count']}")
        if status['specialists']['list']:
            for spec in status['specialists']['list']:
                print(f"  - {spec}")
        else:
            print("  (none)")
        
        print(f"\nIntent Types Logged: {status['logs']['intent_types']}")
        print(f"Total Queries Logged: {status['logs']['total_queries']}\n")