#!/bin/bash

# Add all changes
git add -A

# Commit with detailed message
git commit -m "feat: Add comprehensive enhanced trade reporting system

- Add detailed trade price tracking (entry/exit/stop loss)
- Implement professional report generation (HTML/Markdown/JSON)
- Create standardized reporting framework with themes
- Add stop loss analysis and risk management metrics
- Include MAE/MFE tracking and trade duration analysis
- Generate interactive dashboards with Plotly visualizations
- Add comprehensive documentation and examples
- Update README with enhanced reporting features

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to main branch
git push origin main

echo "✅ Changes committed and pushed successfully!"