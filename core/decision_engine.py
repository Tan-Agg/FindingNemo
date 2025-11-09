# from config import QUERY_THRESHOLD

# class DecisionEngine:
#     """
#     SINGLE RESPONSIBILITY: Decide whether to train a new specialist
#     Checks: threshold, bottleneck analysis, credits
#     """
    
#     def __init__(self):
#         self.threshold = QUERY_THRESHOLD  # Default: 5
        
#         # Hardcoded for MVP - high-value intents worth training
#         self.high_value_intents = [
#             "sql_generation",
#             "code_review",
#             "unit_test_generation",
#             "data_analysis",
#             "text_summarization"
#         ]
        
#         # Training budget
#         self.training_cost = 26  # dollars
#         self.credits_available = 60  # Will be dynamic later
    
#     def check_threshold(self, count):
#         """
#         Check if query count meets threshold
        
#         Args:
#             count (int): Number of queries logged
            
#         Returns:
#             bool: True if threshold met
#         """
#         return count >= self.threshold
    
#     def check_bottleneck(self, intent_label, count):
#         """
#         Check if intent is worth training
#         For MVP: hardcoded high-value intents
        
#         Args:
#             intent_label (str): Intent label
#             count (int): Query count
            
#         Returns:
#             dict: {
#                 "approved": bool,
#                 "reason": str,
#                 "estimated_savings": float (optional)
#             }
#         """
#         # Check if it's a high-value intent
#         if intent_label not in self.high_value_intents:
#             return {
#                 "approved": False,
#                 "reason": f"Intent '{intent_label}' not in high-value list"
#             }
        
#         # Simple ROI calculation (for demo)
#         # Assume: generalist costs $0.00021/query, specialist costs $0.00013/query
#         daily_queries = count  # Assuming these are from 1 day
#         daily_cost_generalist = daily_queries * 0.00021
#         daily_cost_specialist = daily_queries * 0.00013
#         daily_savings = daily_cost_generalist - daily_cost_specialist
        
#         if daily_savings <= 0:
#             return {
#                 "approved": False,
#                 "reason": "No cost savings projected"
#             }
        
#         # Calculate break-even days
#         break_even_days = self.training_cost / daily_savings if daily_savings > 0 else 999999
        
#         # Approve if break-even < 90 days (for demo, we're lenient)
#         if break_even_days <= 90:
#             return {
#                 "approved": True,
#                 "reason": f"Break-even in {break_even_days:.0f} days",
#                 "estimated_savings": daily_savings,
#                 "break_even_days": break_even_days
#             }
#         else:
#             return {
#                 "approved": False,
#                 "reason": f"Break-even too long: {break_even_days:.0f} days"
#             }
    
#     def check_credits(self):
#         """
#         Check if we have enough credits for training
        
#         Returns:
#             bool: True if credits available
#         """
#         return self.credits_available >= self.training_cost
    
#     def make_decision(self, intent_label, count):
#         """
#         Make final decision: train or not?
#         Checks all three conditions: a && b && c
        
#         Args:
#             intent_label (str): Intent label
#             count (int): Query count
            
#         Returns:
#             dict: {
#                 "decision": "TRAIN" or "WAIT",
#                 "reasons": list of strings,
#                 "details": dict with all check results
#             }
#         """
#         reasons = []
#         details = {}
        
#         # Condition A: Threshold
#         a = self.check_threshold(count)
#         details['threshold'] = {
#             "met": a,
#             "count": count,
#             "required": self.threshold
#         }
        
#         if a:
#             reasons.append(f"✅ Threshold met: {count} >= {self.threshold}")
#         else:
#             reasons.append(f"❌ Threshold not met: {count} < {self.threshold}")
        
#         # Condition B: Bottleneck
#         b_result = self.check_bottleneck(intent_label, count)
#         b = b_result['approved']
#         details['bottleneck'] = b_result
        
#         if b:
#             reasons.append(f"✅ Bottleneck approved: {b_result['reason']}")
#         else:
#             reasons.append(f"❌ Bottleneck rejected: {b_result['reason']}")
        
#         # Condition C: Credits
#         c = self.check_credits()
#         details['credits'] = {
#             "available": c,
#             "required": self.training_cost,
#             "remaining": self.credits_available
#         }
        
#         if c:
#             reasons.append(f"✅ Credits available: ${self.credits_available}")
#         else:
#             reasons.append(f"❌ Insufficient credits: ${self.credits_available} < ${self.training_cost}")
        
#         # Final decision: a AND b AND c
#         if a and b and c:
#             decision = "TRAIN"
#             reasons.append("\n🎯 DECISION: Train specialist")
#         else:
#             decision = "WAIT"
#             reasons.append("\n⏸️  DECISION: Wait - conditions not met")
        
#         return {
#             "decision": decision,
#             "reasons": reasons,
#             "details": details
#         }
    
#     def get_training_plan(self, intent_label, description):
#         """
#         Generate a training plan (what will happen)
        
#         Args:
#             intent_label (str): Intent label
#             description (str): Intent description
            
#         Returns:
#             dict: Training plan details
#         """
#         return {
#             "intent_label": intent_label,
#             "description": description,
#             "steps": [
#                 {
#                     "step": 1,
#                     "name": "Generate Training Data",
#                     "tool": "Nemotron-340B",
#                     "details": "Generate 2,000 diverse examples",
#                     "estimated_time": "45 minutes",
#                     "estimated_cost": "$0.84"
#                 },
#                 {
#                     "step": 2,
#                     "name": "Quality Filtering",
#                     "tool": "Nemotron-70B-Reward",
#                     "details": "Score and filter examples",
#                     "estimated_time": "12 minutes",
#                     "estimated_cost": "$0.08"
#                 },
#                 {
#                     "step": 3,
#                     "name": "Fine-Tune Specialist",
#                     "tool": "NeMo",
#                     "details": "Train 8B model on filtered data",
#                     "estimated_time": "6 hours",
#                     "estimated_cost": "$25.00"
#                 },
#                 {
#                     "step": 4,
#                     "name": "Deploy Specialist",
#                     "tool": "NIM",
#                     "details": "Deploy as microservice",
#                     "estimated_time": "8 minutes",
#                     "estimated_cost": "$0.00"
#                 }
#             ],
#             "total_cost": "$25.92",
#             "total_time": "~7 hours",
#             "expected_improvement": {
#                 "latency": "10x faster (2.1s → 0.2s)",
#                 "cost": "100x cheaper per query",
#                 "accuracy": "Maintained (95%+)"
#             }
#         }
from config import QUERY_THRESHOLD

class DecisionEngine:
    """
    SINGLE RESPONSIBILITY: Decide whether to train a new specialist
    Checks: threshold, bottleneck analysis, credits
    """
    
    def __init__(self):
        self.threshold = QUERY_THRESHOLD  # Default: 5
        
        # Training budget
        self.training_cost = 26  # dollars
        self.credits_available = 60  # Will be dynamic later
    
    def check_threshold(self, count):
        """
        Check if query count meets threshold
        
        Args:
            count (int): Number of queries logged
            
        Returns:
            bool: True if threshold met
        """
        return count >= self.threshold
    
    def check_bottleneck(self, intent_label, count):
        """
        Check if intent is worth training
        For MVP: ALWAYS APPROVED (condition B always met)
        
        Args:
            intent_label (str): Intent label
            count (int): Query count
            
        Returns:
            dict: Always returns approved=True
        """
        # Simple ROI calculation (for demo)
        # Assume: generalist costs $0.00021/query, specialist costs $0.00013/query
        daily_queries = count
        daily_cost_generalist = daily_queries * 0.00021
        daily_cost_specialist = daily_queries * 0.00013
        daily_savings = daily_cost_generalist - daily_cost_specialist
        
        # Calculate break-even days
        break_even_days = self.training_cost / daily_savings if daily_savings > 0 else 999999
        
        # ALWAYS APPROVE - condition B is always met
        return {
            "approved": True,
            "reason": f"Auto-approved (Break-even in {break_even_days:.0f} days)",
            "estimated_savings": daily_savings,
            "break_even_days": break_even_days
        }
    
    def check_credits(self):
        """
        Check if we have enough credits for training
        For MVP: ALWAYS TRUE (condition C always met)
        
        Returns:
            bool: Always True
        """
        return True  # Always have credits for MVP
    
    def make_decision(self, intent_label, count):
        """
        Make final decision: train or not?
        Checks all three conditions: a && b && c
        Where b=True, c=True always, only a depends on input from log
        
        Args:
            intent_label (str): Intent label
            count (int): Query count from log file
            
        Returns:
            dict: {
                "decision": "TRAIN" or "WAIT",
                "reasons": list of strings,
                "details": dict with all check results
            }
        """
        reasons = []
        details = {}
        
        # Condition A: Threshold (DEPENDS ON LOG INPUT)
        a = self.check_threshold(count)
        details['threshold'] = {
            "met": a,
            "count": count,
            "required": self.threshold
        }
        
        if a:
            reasons.append(f"✅ Threshold met: {count} >= {self.threshold}")
        else:
            reasons.append(f"❌ Threshold not met: {count} < {self.threshold}")
        
        # Condition B: Bottleneck (ALWAYS TRUE)
        b_result = self.check_bottleneck(intent_label, count)
        b = b_result['approved']  # Always True
        details['bottleneck'] = b_result
        reasons.append(f"✅ Bottleneck approved: {b_result['reason']}")
        
        # Condition C: Credits (ALWAYS TRUE)
        c = self.check_credits()  # Always True
        details['credits'] = {
            "available": c,
            "required": self.training_cost,
            "remaining": self.credits_available
        }
        reasons.append(f"✅ Credits available: ${self.credits_available}")
        
        # Final decision: a AND b AND c
        # Since b=True, c=True always, decision only depends on a (threshold)
        if a:  # Since b and c are always True
            decision = "TRAIN"
            reasons.append("\n🎯 DECISION: Train specialist")
        else:
            decision = "WAIT"
            reasons.append("\n⏸️  DECISION: Wait - threshold not met")
        
        return {
            "decision": decision,
            "reasons": reasons,
            "details": details
        }
    
    def get_training_plan(self, intent_label, description):
        """
        Generate a training plan (what will happen)
        
        Args:
            intent_label (str): Intent label
            description (str): Intent description
            
        Returns:
            dict: Training plan details
        """
        return {
            "intent_label": intent_label,
            "description": description,
            "steps": [
                {
                    "step": 1,
                    "name": "Generate Training Data",
                    "tool": "Nemotron-340B",
                    "details": "Generate 2,000 diverse examples",
                    "estimated_time": "45 minutes",
                    "estimated_cost": "$0.84"
                },
                {
                    "step": 2,
                    "name": "Quality Filtering",
                    "tool": "Nemotron-70B-Reward",
                    "details": "Score and filter examples",
                    "estimated_time": "12 minutes",
                    "estimated_cost": "$0.08"
                },
                {
                    "step": 3,
                    "name": "Fine-Tune Specialist",
                    "tool": "NeMo",
                    "details": "Train 8B model on filtered data",
                    "estimated_time": "6 hours",
                    "estimated_cost": "$25.00"
                },
                {
                    "step": 4,
                    "name": "Deploy Specialist",
                    "tool": "NIM",
                    "details": "Deploy as microservice",
                    "estimated_time": "8 minutes",
                    "estimated_cost": "$0.00"
                }
            ],
            "total_cost": "$25.92",
            "total_time": "~7 hours",
            "expected_improvement": {
                "latency": "10x faster (2.1s → 0.2s)",
                "cost": "100x cheaper per query",
                "accuracy": "Maintained (95%+)"
            }
        }

