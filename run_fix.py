import subprocess
import sys
import os

# Change to the project directory
os.chdir("/workspaces/Backtest_Suite")

# Run the fix script
try:
    result = subprocess.run([sys.executable, "fix_shell_and_commit.py"], 
                          capture_output=True, text=True, timeout=120)
    
    print("Exit code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
except subprocess.TimeoutExpired:
    print("Script timed out")
except Exception as e:
    print(f"Error: {e}")