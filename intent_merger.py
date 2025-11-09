from sentence_transformers import SentenceTransformer
import numpy as np
import json

class IntentMerger:
    """
    Detect and merge duplicate intents that are semantically similar
    """
    
    def __init__(self, embedding_model=None):
        if embedding_model:
            self.model = embedding_model
        else:
            print("Loading embedding model for intent merging...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.merge_threshold = 0.30  # 75% similarity = same intent
    
    def calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        emb1 = self.model.encode(text1, normalize_embeddings=True)
        emb2 = self.model.encode(text2, normalize_embeddings=True)
        similarity = np.dot(emb1, emb2)
        return similarity
    
    def find_duplicates(self, intents):
        """
        Find duplicate intents based on semantic similarity
        
        Args:
            intents: list of dicts with 'intent_label' and 'description'
            
        Returns:
            list of groups of duplicate intents
        """
        if len(intents) <= 1:
            return []
        
        duplicates = []
        processed = set()
        
        for i, intent1 in enumerate(intents):
            if i in processed:
                continue
            
            group = [intent1]
            
            for j, intent2 in enumerate(intents[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # Compare descriptions
                sim = self.calculate_similarity(
                    intent1['description'],
                    intent2['description']
                )
                
                if sim >= self.merge_threshold:
                    group.append(intent2)
                    processed.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
                processed.add(i)
        
        return duplicates
    
    def merge_intents_in_logs(self, query_logger, dry_run=True):
        """
        Merge duplicate intents in query logs
        
        Args:
            query_logger: QueryLogger instance
            dry_run: If True, only show what would be merged
            
        Returns:
            dict with merge actions
        """
        logs = query_logger.get_all_logs()
        
        if not logs:
            return {'merged': 0, 'groups': [], 'dry_run': dry_run}
        
        # Convert to list format - handle different log structures
        intent_list = []
        for label, data in logs.items():
            try:
                # Handle both dict and list formats
                if isinstance(data, dict):
                    # Try both 'canonical_description' and 'description'
                    description = data.get('canonical_description') or data.get('description') or label
                    intent_list.append({
                        'intent_label': label,
                        'description': description,
                        'count': data.get('count', len(data.get('queries', []))),
                        'queries': data.get('queries', [])
                    })
                else:
                    # If data is a list of queries
                    intent_list.append({
                        'intent_label': label,
                        'description': label,  # Use label as description
                        'count': len(data),
                        'queries': data
                    })
            except Exception as e:
                print(f"⚠️  Warning: Skipping intent '{label}' due to error: {e}")
                continue
        
        # Find duplicates
        duplicate_groups = self.find_duplicates(intent_list)
        
        if not duplicate_groups:
            return {'merged': 0, 'groups': []}
        
        merge_actions = []
        
        for group in duplicate_groups:
            # Choose primary intent (one with most queries)
            primary = max(group, key=lambda x: x['count'])
            others = [x for x in group if x != primary]
            
            action = {
                'primary': primary['intent_label'],
                'merge_into': [x['intent_label'] for x in others],
                'total_count': sum(x['count'] for x in group),
                'similarity_scores': []
            }
            
            # Calculate similarities
            for other in others:
                sim = self.calculate_similarity(
                    primary['description'],
                    other['description']
                )
                action['similarity_scores'].append({
                    'intent': other['intent_label'],
                    'similarity': round(sim, 3)
                })
            
            merge_actions.append(action)
            
            # Actually merge if not dry run
            if not dry_run:
                # Merge all queries into primary
                for other in others:
                    primary['queries'].extend(other['queries'])
                    primary['count'] += other['count']
                    # Delete the duplicate
                    del logs[other['intent_label']]
                
                # Update the primary
                logs[primary['intent_label']] = {
                    'canonical_description': primary['description'],
                    'description': primary['description'],  # Keep both for compatibility
                    'count': primary['count'],
                    'queries': primary['queries'],
                    'first_seen': logs[primary['intent_label']].get('first_seen', ''),
                    'last_seen': logs[primary['intent_label']].get('last_seen', '')
                }
                
                # Save updated logs
                query_logger.logs = logs
                query_logger.save()
        
        return {
            'merged': len(duplicate_groups),
            'groups': merge_actions,
            'dry_run': dry_run
        }
    
    def print_merge_report(self, merge_result):
        """Pretty print merge results"""
        if merge_result['merged'] == 0:
            print("\n✅ No duplicate intents found!")
            return
        
        print(f"\n🔍 Found {merge_result['merged']} duplicate intent group(s):\n")
        
        for i, action in enumerate(merge_result['groups'], 1):
            print(f"Group {i}:")
            print(f"  ✅ Keep: '{action['primary']}' (Total: {action['total_count']} queries)")
            print(f"  🔄 Merge from:")
            
            for merge_from in action['similarity_scores']:
                print(f"      - '{merge_from['intent']}' (Similarity: {merge_from['similarity']})")
            
            print()
        
        if merge_result['dry_run']:
            print("⚠️  This was a DRY RUN - no changes made")
            print("   Run with dry_run=False to actually merge\n")
        else:
            print("✅ Intents merged successfully!\n")


# Add this to your main.py
def merge_duplicate_intents(agent, dry_run=True):
    """
    Helper function to merge duplicate intents
    
    Usage:
        merge_duplicate_intents(agent, dry_run=True)  # Preview
        merge_duplicate_intents(agent, dry_run=False) # Actually merge
    """
    print("\n" + "="*60)
    print("CHECKING FOR DUPLICATE INTENTS")
    print("="*60)
    
    merger = IntentMerger(agent.embedding_service.model)
    result = merger.merge_intents_in_logs(agent.query_logger, dry_run=dry_run)
    merger.print_merge_report(result)
    
    return result


# ============================================================
# ADD THIS TO THE BOTTOM OF YOUR main.py (before final status)
# ============================================================

if __name__ == "__main__":
    # ... your existing code ...
    
    # After running test queries, check for duplicates:
    print("\n" + "="*60)
    print("CHECKING FOR DUPLICATE INTENTS")
    print("="*60)
    
    merger = IntentMerger(agent.embedding_service.model)
    
    # First, preview what would be merged
    print("\n📋 DRY RUN - Preview of merge actions:\n")
    result = merger.merge_intents_in_logs(agent.query_logger, dry_run=True)
    merger.print_merge_report(result)
    
    # If duplicates found, ask to merge
    if result['merged'] > 0:
        response = input("\n❓ Merge these intents? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\n🔄 Merging intents...")
            result = merger.merge_intents_in_logs(agent.query_logger, dry_run=False)
            merger.print_merge_report(result)
            print("✅ Logs updated!")
        else:
            print("\n❌ Merge cancelled")
    
    # Then show final status
    print("\n" + "="*60)
    print("FINAL STATUS AFTER MERGE")
    print("="*60)
    agent.print_status()