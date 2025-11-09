import json
from datetime import datetime

class QueryLogger:
    """
    SINGLE RESPONSIBILITY: Log queries and track counts
    """
    
    def __init__(self, log_file='data/query_logs.json'):
        self.log_file = log_file
        self.logs = {}
        self.load()
    
    def load(self):
        """Load logs from disk"""
        try:
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
            print(f"✅ Loaded logs for {len(self.logs)} intent types")
        except FileNotFoundError:
            print(f"⚠️  Log file not found, creating new")
            self.logs = {}
            self.save()
    
    def save(self):
        """Save logs to disk"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def log_query(self, intent_label, intent_description, user_prompt):
        """
        Log a query for an intent
        
        Args:
            intent_label (str): Intent label (e.g., "sql_generation")
            intent_description (str): Description of intent
            user_prompt (str): User's original query
        """
        timestamp = datetime.now().isoformat()
        
        if intent_label in self.logs:
            # Intent exists - increment count
            self.logs[intent_label]['count'] += 1
            self.logs[intent_label]['last_seen'] = timestamp
            self.logs[intent_label]['queries'].append({
                "prompt": user_prompt,
                "timestamp": timestamp
            })
        else:
            # New intent - create entry
            self.logs[intent_label] = {
                "count": 1,
                "canonical_description": intent_description,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "queries": [{
                    "prompt": user_prompt,
                    "timestamp": timestamp
                }]
            }
        
        self.save()
        print(f"✅ Logged query for '{intent_label}' (count: {self.logs[intent_label]['count']})")
    
    def get_count(self, intent_label):
        """Get count for specific intent"""
        if intent_label in self.logs:
            return self.logs[intent_label]['count']
        return 0
    
    def get_bottlenecks(self, threshold=5):
        """
        Get intents that have crossed the threshold
        
        Args:
            threshold (int): Minimum count to be considered bottleneck
            
        Returns:
            list: List of intents with count >= threshold
        """
        bottlenecks = []
        
        for intent_label, data in self.logs.items():
            if data['count'] >= threshold:
                bottlenecks.append({
                    "intent_label": intent_label,
                    "description": data['canonical_description'],
                    "count": data['count']
                })
        
        return bottlenecks
    
    def delete_log(self, intent_label):
        """Delete log entry (when specialist is created)"""
        if intent_label in self.logs:
            del self.logs[intent_label]
            self.save()
            print(f"✅ Deleted logs for '{intent_label}'")
            return True
        return False
    
    def get_all_logs(self):
        """Return all logs"""
        return self.logs