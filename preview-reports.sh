#!/bin/bash
# Backtest Suite Reports Preview Server
# Usage: ./preview-reports.sh

echo "🚀 Starting Backtest Suite Reports Preview Server..."
echo "📊 Reports will be available at:"
echo "   - http://localhost:8000"
echo "   - Or check the PORTS tab for the forwarded URL"
echo ""
echo "📁 Available reports:"
echo "   - Main Dashboard: http://localhost:8000/index.html"
echo "   - ML Performance: http://localhost:8000/performance/ML_RandomForest_performance_report.html"
echo "   - Strategy Comparison: http://localhost:8000/comparison/strategy_comparison_report.html"
echo "   - Executive Summary: http://localhost:8000/summary/executive_summary_report.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo "-----------------------------------"

cd /workspaces/Backtest_Suite/reports
python -m http.server 8000 --bind 0.0.0.0