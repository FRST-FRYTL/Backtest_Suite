#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

# Change to project directory
os.chdir("/workspaces/Backtest_Suite")

print("🚀 Starting commit process...")

# Step 1: Stage all changes
try:
    result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"❌ Failed to stage changes: {result.stderr}")
        exit(1)
    print("✅ Staged all changes")
except Exception as e:
    print(f"❌ Error staging changes: {e}")
    exit(1)

# Step 2: Commit with message
commit_message = """feat: Add comprehensive enhanced trade reporting system

- Add detailed trade price tracking (entry/exit/stop loss)
- Implement professional report generation (HTML/Markdown/JSON)
- Create standardized reporting framework with themes
- Add stop loss analysis and risk management metrics
- Include MAE/MFE tracking and trade duration analysis
- Generate interactive dashboards with Plotly visualizations
- Add comprehensive documentation and examples
- Update README with enhanced reporting features
- Fix shell snapshot issue caused by Claude Flow hooks

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

try:
    result = subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"❌ Failed to commit: {result.stderr}")
        exit(1)
    print("✅ Committed changes")
    print("Commit output:", result.stdout)
except Exception as e:
    print(f"❌ Error committing: {e}")
    exit(1)

# Step 3: Push to main
try:
    result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"❌ Failed to push: {result.stderr}")
        exit(1)
    print("✅ Pushed to main branch")
    print("Push output:", result.stdout)
except Exception as e:
    print(f"❌ Error pushing: {e}")
    exit(1)

print("\n🎉 SUCCESS! Enhanced trade reporting system committed and pushed!")
print("\n📊 Summary of changes:")
print("  ✅ Enhanced trade reporting with entry/exit/stop loss prices")
print("  ✅ Professional HTML/Markdown/JSON report generation")
print("  ✅ Interactive dashboards with Plotly visualizations")
print("  ✅ Comprehensive documentation and examples")
print("  ✅ Updated README with new features")
print("  ✅ Fixed shell snapshot issue")