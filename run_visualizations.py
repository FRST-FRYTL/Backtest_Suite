#!/usr/bin/env python3
"""Run comprehensive trading visualizations"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the visualization module
from visualization.comprehensive_trading_dashboard import main

if __name__ == "__main__":
    main()