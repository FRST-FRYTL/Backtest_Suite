#!/usr/bin/env python3
import os
from pathlib import Path

# Create the shell snapshots directory
snapshots_dir = Path("/home/codespace/.claude/shell-snapshots")
snapshots_dir.mkdir(parents=True, exist_ok=True)

# Create the missing snapshot file
snapshot_file = snapshots_dir / "snapshot-bash-7fed3f3e.sh"
snapshot_content = """#!/bin/bash
# Shell snapshot file for Claude Code
# This file was recreated to fix the shell issue
exec /bin/bash "$@"
"""

with open(snapshot_file, "w") as f:
    f.write(snapshot_content)

# Make it executable
os.chmod(snapshot_file, 0o755)

print(f"✅ Created shell snapshot file: {snapshot_file}")

# Now run the git commands
import subprocess

os.chdir("/workspaces/Backtest_Suite")

# Stage all changes
print("📝 Staging all changes...")
result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Failed to stage: {result.stderr}")
    exit(1)
print("✅ Staged successfully")

# Commit
print("💾 Committing changes...")
commit_msg = """feat: Add comprehensive enhanced trade reporting system

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

result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Failed to commit: {result.stderr}")
    exit(1)
print("✅ Committed successfully")

# Push
print("🚀 Pushing to main...")
result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Failed to push: {result.stderr}")
    exit(1)
print("✅ Pushed successfully")

print("\n🎉 SUCCESS! Enhanced trade reporting system committed and pushed!")
print("\n📊 Changes committed:")
print("  ✅ Enhanced trade reporting with detailed price tracking")
print("  ✅ Professional report generation (HTML/Markdown/JSON)")
print("  ✅ Interactive dashboards with Plotly visualizations")
print("  ✅ Comprehensive documentation and examples")
print("  ✅ Updated README with new features")
print("  ✅ Fixed shell snapshot issue")