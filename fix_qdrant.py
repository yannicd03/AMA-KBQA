#!/usr/bin/env python3
"""
Automated fix script for kqapro_server.py
Applies all Qdrant client and SPARQL access fixes
"""

import re
import sys
from pathlib import Path


def apply_fixes(file_path: str) -> tuple[str, list[str]]:
    """
    Apply all fixes to the file content.
    
    Returns:
        (fixed_content, list_of_changes_made)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes = []
    original_content = content
    
    # Fix 1: Replace .search() with .query_points()
    # Pattern: client.search(collection_name=..., query_vector=vector, ...)
    search_pattern = r'([\w.]+)\.search\(\s*collection_name=([^,]+),\s*query_vector=([^,]+),\s*limit=([^,]+),\s*with_payload=True\s*\)'
    search_replacement = r'\1.query_points(collection_name=\2, query=\3, limit=\4, with_payload=True).points'
    
    new_content = re.sub(search_pattern, search_replacement, content)
    if new_content != content:
        changes.append("✅ Replaced .search() with .query_points() and added .points accessor")
        content = new_content
    
    # Fix 2: Replace query_vector= with query= (in case pattern above didn't catch all)
    new_content = content.replace('query_vector=', 'query=')
    if new_content != content:
        changes.append("✅ Replaced query_vector= with query=")
        content = new_content
    
    # Fix 3: Fix SPARQL access patterns
    # Pattern 1: ctx.qdrant.sparql → app_context.sparql
    patterns_to_fix = [
        (r'ctx\.qdrant\.sparql', 'app_context.sparql', 'ctx.qdrant.sparql → app_context.sparql'),
        (r'app_context\.qdrant\.sparql', 'app_context.sparql', 'app_context.qdrant.sparql → app_context.sparql'),
    ]
    
    for pattern, replacement, description in patterns_to_fix:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            count = len(re.findall(pattern, content))
            changes.append(f"✅ Fixed {count} occurrences of {description}")
            content = new_content
    
    # Fix 4: Ensure consistent variable naming in ExploreNeighborhood
    # Look for the function definition and fix it
    explore_neighborhood_pattern = r'def ExploreNeighborhood\(.*?\):\s*""".*?"""(.*?)(?=\n@mcp\.tool|\nclass |\ndef |\Z)'
    
    def fix_explore_neighborhood(match):
        func_body = match.group(1)
        
        # If it has 'ctx: AppContext' but uses 'app_context' later, change to app_context
        if 'ctx: AppContext = context.request_context.lifespan_context' in func_body:
            if 'app_context.qdrant' in func_body or 'app_context.sparql' in func_body:
                # Change ctx to app_context
                func_body = func_body.replace(
                    'ctx: AppContext = context.request_context.lifespan_context',
                    'app_context: AppContext = context.request_context.lifespan_context'
                )
                func_body = func_body.replace('ctx.sparql', 'app_context.sparql')
                func_body = func_body.replace('ctx.qdrant', 'app_context.qdrant')
                func_body = func_body.replace('ctx.embedding_client', 'app_context.embedding_client')
        
        return 'def ExploreNeighborhood' + match.group(0).split('def ExploreNeighborhood')[1].split('"""')[0] + '"""' + func_body
    
    new_content = re.sub(explore_neighborhood_pattern, fix_explore_neighborhood, content, flags=re.DOTALL)
    if new_content != content:
        changes.append("✅ Fixed variable naming in ExploreNeighborhood")
        content = new_content
    
    # Fix 5: Remove debugging code in ExploreNeighborhood
    debug_pattern = r'\s*# --- DEBUGGING START ---.*?# --- DEBUGGING END ---\s*'
    new_content = re.sub(debug_pattern, '\n', content, flags=re.DOTALL)
    if new_content != content:
        changes.append("✅ Removed debugging code")
        content = new_content
    
    # Fix 6: Remove incorrect fallback logic in ExploreNeighborhood
    fallback_pattern = r'except AttributeError:.*?raise RuntimeError\(.*?\)'
    new_content = re.sub(fallback_pattern, 
        '''except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        return NeighborhoodResponse(
            base_node=base_node_id,
            verified_match=None,
            candidates_checked=[],
            status=f"Qdrant error: {str(e)}"
        )''', 
        content, flags=re.DOTALL)
    if new_content != content:
        changes.append("✅ Fixed error handling in ExploreNeighborhood")
        content = new_content
    
    if content == original_content:
        changes.append("ℹ️  No changes needed - file is already correct")
    
    return content, changes


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_qdrant_issues.py <path_to_kqapro_server.py>")
        print("\nExample: python fix_qdrant_issues.py ./ama_kbqa/server/kqapro_server.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"🔧 Applying fixes to: {file_path}")
    print("=" * 70)
    
    try:
        fixed_content, changes = apply_fixes(file_path)
        
        # Create backup
        backup_path = f"{file_path}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as original:
                f.write(original.read())
        print(f"✅ Created backup: {backup_path}")
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("\n📋 Changes applied:")
        for change in changes:
            print(f"  {change}")
        
        print("\n" + "=" * 70)
        print("✅ Fixes applied successfully!")
        print(f"📄 Original file backed up to: {backup_path}")
        print("\n⚠️  IMPORTANT NEXT STEPS:")
        print("  1. Review the changes in your editor")
        print("  2. Test the server with: python -m ama_kbqa.server.kqapro_server")
        print("  3. If there are issues, restore from backup")
        print("  4. Make sure qdrant-client >= 1.13.0 is installed")
        
    except Exception as e:
        print(f"\n❌ Error applying fixes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()