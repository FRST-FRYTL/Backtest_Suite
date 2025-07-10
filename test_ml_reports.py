"""
Standalone test for ML Report Generator
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Import only the report generator without going through src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import
import importlib.util
spec = importlib.util.spec_from_file_location("report_generator", "src/ml/reports/report_generator.py")
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)
MLReportGenerator = report_module.MLReportGenerator


def test_report_generation():
    """Test basic report generation."""
    print("Testing ML Report Generator...")
    
    # Initialize report generator
    report_gen = MLReportGenerator()
    print(f"✓ Report generator initialized")
    print(f"  Template directory: {report_gen.template_dir}")
    print(f"  Output directory: {report_gen.output_dir}")
    
    # Generate sample data
    features = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E']
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'importance': [0.35, 0.25, 0.20, 0.15, 0.05]
    }, index=features)
    
    # Correlation matrix
    n_features = len(features)
    corr_data = np.random.uniform(-0.8, 0.8, (n_features, n_features))
    np.fill_diagonal(corr_data, 1.0)
    correlation_matrix = pd.DataFrame(corr_data, index=features, columns=features)
    
    # Feature distributions
    feature_distributions = {
        feature: pd.Series(np.random.normal(0, 1, 500))
        for feature in features[:3]
    }
    
    # Generate feature analysis report
    print("\nGenerating Feature Analysis Report...")
    try:
        report_path = report_gen.generate_feature_analysis_report(
            feature_importance=feature_importance,
            correlation_matrix=correlation_matrix,
            feature_distributions=feature_distributions,
            metadata={'test': True, 'timestamp': datetime.now().isoformat()}
        )
        print(f"✓ Feature analysis report generated: {report_path}")
    except Exception as e:
        print(f"✗ Error generating feature analysis report: {e}")
    
    # Test performance dashboard
    print("\nGenerating Performance Dashboard...")
    
    model_metrics = {
        'Model_A': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1_score': 0.825,
            'auc': 0.89
        },
        'Model_B': {
            'accuracy': 0.82,
            'precision': 0.80,
            'recall': 0.79,
            'f1_score': 0.795,
            'auc': 0.86
        }
    }
    
    confusion_matrices = {
        'Model_A': np.array([[80, 20], [15, 85]]),
        'Model_B': np.array([[75, 25], [18, 82]])
    }
    
    # Simple ROC data
    fpr = np.linspace(0, 1, 50)
    roc_data = {
        'Model_A': {
            'fpr': fpr,
            'tpr': 1 - np.exp(-2.5 * fpr),
            'auc': 0.89
        },
        'Model_B': {
            'fpr': fpr,
            'tpr': 1 - np.exp(-2 * fpr),
            'auc': 0.86
        }
    }
    
    # Profit curves
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    profit_curves = {
        'Model_A': pd.DataFrame({
            'cumulative_profit': np.cumsum(np.random.normal(0.001, 0.01, len(dates)))
        }, index=dates),
        'Model_B': pd.DataFrame({
            'cumulative_profit': np.cumsum(np.random.normal(0.0005, 0.012, len(dates)))
        }, index=dates)
    }
    
    try:
        report_path = report_gen.generate_performance_dashboard(
            model_metrics=model_metrics,
            confusion_matrices=confusion_matrices,
            roc_data=roc_data,
            profit_curves=profit_curves,
            metadata={'test': True}
        )
        print(f"✓ Performance dashboard generated: {report_path}")
    except Exception as e:
        print(f"✗ Error generating performance dashboard: {e}")
    
    print("\n✅ Test completed!")
    print(f"\nReports are available in: {report_gen.output_dir}")
    
    # List generated reports
    import glob
    reports = glob.glob(os.path.join(report_gen.output_dir, '*.html'))
    if reports:
        print("\nGenerated reports:")
        for report in reports:
            print(f"  - {os.path.basename(report)}")


if __name__ == '__main__':
    test_report_generation()