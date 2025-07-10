# How to Preview HTML in GitHub Codespaces

## The Codespaces Way 🚀

In GitHub Codespaces, previewing HTML works differently than in local VS Code. Here are the correct methods:

## Method 1: Simple HTTP Server (Recommended)

### Option A: Using Python (Simplest)
```bash
# Navigate to your reports directory
cd /workspaces/Backtest_Suite/reports

# Start a simple HTTP server
python -m http.server 8000
```

Then:
1. Look for the **"Open in Browser"** popup notification
2. Or go to the **PORTS** tab in the terminal panel
3. Click on the port URL to open in browser

### Option B: Using Live Server Extension Command
```bash
# In the terminal, run:
npx live-server reports --port=5500
```

## Method 2: Using the Ports Panel

1. Open the **PORTS** tab (next to TERMINAL, PROBLEMS, OUTPUT)
2. Click **"Add Port"**
3. Enter `8000` (or any port)
4. Start your server on that port
5. Click the globe icon 🌐 to open in browser

## Method 3: Command Palette Preview

1. Open your HTML file
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
3. Type: **"Simple Browser: Show"**
4. Enter the URL: `http://localhost:8000/index.html`

## Method 4: VS Code Preview (Built-in)

1. Open the HTML file
2. Press `Ctrl+Shift+V` to open preview
3. Note: This shows raw HTML preview, not served via HTTP

## Quick Start Script

Create this helper script for easy previewing:

```bash
#!/bin/bash
# Save as: preview-reports.sh

echo "Starting preview server for Backtest Suite reports..."
cd /workspaces/Backtest_Suite/reports
python -m http.server 8000 --bind 0.0.0.0
```

## Why Codespaces is Different

- **Security**: Codespaces runs in a container with port forwarding
- **No Direct File Access**: Can't open local files directly in browser
- **Port Forwarding**: All preview must go through forwarded ports
- **Automatic HTTPS**: Codespaces automatically secures your preview URLs

## Pro Tips for Codespaces

1. **Port Visibility**: 
   - Private: Only you can access
   - Public: Anyone with the URL can access
   - Set in the PORTS panel

2. **Multiple Servers**: You can run multiple servers on different ports

3. **Persistent URLs**: Forwarded URLs remain the same for your Codespace session

4. **Auto-forwarding**: Codespaces automatically detects and forwards common ports (3000, 5000, 8000, 8080)

## Your Report URLs

Once you start the server on port 8000:
- Main Dashboard: `http://localhost:8000/index.html`
- ML Report: `http://localhost:8000/performance/ML_RandomForest_performance_report.html`
- Comparison: `http://localhost:8000/comparison/strategy_comparison_report.html`
- Summary: `http://localhost:8000/summary/executive_summary_report.html`

The localhost URLs will automatically redirect to the Codespaces forwarded URL!