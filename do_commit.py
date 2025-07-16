#!/usr/bin/env python3
"""
Simple Python script to commit and push changes
"""
import subprocess
import os

def run_command(command):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd='/workspaces/Backtest_Suite')
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        print(f"Success: {command}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        return False

def main():
    print("🔄 Starting git commit and push process...")
    
    # Change to project directory
    os.chdir('/workspaces/Backtest_Suite')
    
    # Add all changes
    print("📁 Adding all changes...")
    if not run_command("git add -A"):
        return False
    
    # Commit with message
    print("💾 Committing changes...")
    commit_message = """feat: Add comprehensive enhanced trade reporting system

- Add detailed trade price tracking (entry/exit/stop loss)
- Implement professional report generation (HTML/Markdown/JSON)
- Create standardized reporting framework with themes
- Add stop loss analysis and risk management metrics
- Include MAE/MFE tracking and trade duration analysis
- Generate interactive dashboards with Plotly visualizations
- Add comprehensive documentation and examples
- Update README with enhanced reporting features

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
    
    if not run_command(f'git commit -m "{commit_message}"'):
        return False
    
    # Push to main
    print("🚀 Pushing to main branch...")
    if not run_command("git push origin main"):
        return False
    
    print("✅ All changes committed and pushed successfully!")
    return True

if __name__ == "__main__":
    main()