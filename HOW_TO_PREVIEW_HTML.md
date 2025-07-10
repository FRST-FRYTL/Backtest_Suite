# How to Preview HTML Reports in VS Code

## Quick Start

### Method 1: Live Server Extension (Recommended)
1. **Open any HTML report file** (e.g., `/reports/performance/ML_RandomForest_performance_report.html`)
2. **Right-click** on the file in the editor
3. Select **"Open with Live Server"**
4. Your default browser will open with live reload enabled

Alternative: Click the **"Go Live"** button in the VS Code status bar (bottom right)

### Method 2: VS Code Built-in Preview
1. Open any HTML file
2. Press `Ctrl+Shift+V` (Windows/Linux) or `Cmd+Shift+V` (Mac)
3. Or click the preview icon (📄) in the top right corner

### Method 3: Command Palette
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Live Server: Open with Live Server"
3. Press Enter

## Configuration Complete! 🎉

I've configured VS Code with:

### 1. **Live Server Settings** (`.vscode/settings.json`)
- Port: 5500
- Auto-refresh on file changes
- Root directory set to `/reports`
- HTML files associated with Live Preview

### 2. **Recommended Extensions** (`.vscode/extensions.json`)
- Live Server
- Python tools
- Jupyter support
- Data preview extensions

### 3. **Launch Configurations** (`.vscode/launch.json`)
- Python debugging
- Backtest runner
- Report generator
- Chrome debugger for Live Server

## Viewing Your Reports

### Main Dashboard
- Open: `/reports/index.html`
- This is your central hub for all reports

### Individual Reports
- Performance reports: `/reports/performance/*.html`
- Strategy comparison: `/reports/comparison/strategy_comparison_report.html`
- Executive summary: `/reports/summary/executive_summary_report.html`

## Tips

1. **Auto-reload**: Files automatically refresh when you save changes
2. **Multiple files**: You can preview multiple HTML files simultaneously
3. **Network access**: Share your preview with others on the same network using your IP:5500
4. **Mobile preview**: Access from your phone using `http://[your-ip]:5500`

## Keyboard Shortcuts

- `Alt+L Alt+O`: Open with Live Server
- `Alt+L Alt+C`: Stop Live Server
- `Ctrl+Shift+V`: VS Code preview
- `Ctrl+K V`: Open preview to the side

## Troubleshooting

If Live Server doesn't work:
1. Make sure the extension is installed: `code --install-extension ms-vscode.live-server`
2. Check if port 5500 is available
3. Try a different port in settings
4. Restart VS Code

Enjoy viewing your backtesting reports! 📊