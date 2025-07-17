"""
Comprehensive tests for ML clustering functionality in SuperTrend AI.

This module provides complete test coverage for the K-means clustering optimizer
used in the SuperTrend AI indicator for parameter optimization.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.ml.clustering.kmeans_optimizer import KMeansOptimizer
from src.ml.clustering import ClusteringError


class TestKMeansOptimizer:
    """Comprehensive tests for KMeansOptimizer class."""
    
    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance data for clustering."""
        np.random.seed(42)
        
        # Create realistic performance data with distinct clusters
        # Good performers
        good_performers = pd.DataFrame({
            'sharpe_ratio': np.random.normal(2.0, 0.3, 30),
            'max_drawdown': np.random.normal(-0.08, 0.02, 30),
            'win_rate': np.random.normal(0.65, 0.05, 30),
            'profit_factor': np.random.normal(2.5, 0.5, 30),
            'total_trades': np.random.normal(100, 20, 30)
        })
        
        # Average performers
        avg_performers = pd.DataFrame({
            'sharpe_ratio': np.random.normal(1.0, 0.2, 40),
            'max_drawdown': np.random.normal(-0.15, 0.03, 40),
            'win_rate': np.random.normal(0.55, 0.05, 40),
            'profit_factor': np.random.normal(1.5, 0.3, 40),
            'total_trades': np.random.normal(80, 15, 40)
        })
        
        # Poor performers
        poor_performers = pd.DataFrame({
            'sharpe_ratio': np.random.normal(0.2, 0.2, 30),
            'max_drawdown': np.random.normal(-0.25, 0.05, 30),
            'win_rate': np.random.normal(0.45, 0.05, 30),
            'profit_factor': np.random.normal(0.8, 0.2, 30),
            'total_trades': np.random.normal(60, 10, 30)
        })
        
        # Combine all data
        data = pd.concat([good_performers, avg_performers, poor_performers], ignore_index=True)
        
        # Add parameter columns
        data['atr_period'] = np.random.choice([7, 10, 14, 20], len(data))
        data['multiplier'] = np.random.choice([1.5, 2.0, 2.5, 3.0], len(data))
        
        return data
    
    def test_initialization_default(self):
        """Test KMeansOptimizer initialization with default parameters."""
        optimizer = KMeansOptimizer()
        
        assert optimizer.n_clusters == 3
        assert optimizer.max_iter == 300
        assert optimizer.random_state == 42
        assert optimizer.optimize_clusters == True
        assert optimizer.scaler is None
        assert optimizer.model is None
        assert optimizer.cluster_centers_ is None
        assert optimizer.labels_ is None
        assert optimizer.performance_weights == {
            'sharpe_ratio': 0.3,
            'max_drawdown': 0.25,
            'win_rate': 0.2,
            'profit_factor': 0.15,
            'total_trades': 0.1
        }
    
    def test_initialization_custom(self):
        """Test KMeansOptimizer initialization with custom parameters."""
        custom_weights = {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.3,
            'win_rate': 0.2
        }
        
        optimizer = KMeansOptimizer(
            n_clusters=5,
            max_iter=500,
            random_state=123,
            optimize_clusters=False,
            performance_weights=custom_weights
        )
        
        assert optimizer.n_clusters == 5
        assert optimizer.max_iter == 500
        assert optimizer.random_state == 123
        assert optimizer.optimize_clusters == False
        assert optimizer.performance_weights == custom_weights
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Invalid n_clusters
        with pytest.raises(ValueError, match="n_clusters must be positive"):
            KMeansOptimizer(n_clusters=0)
        
        # Invalid max_iter
        with pytest.raises(ValueError, match="max_iter must be positive"):
            KMeansOptimizer(max_iter=0)
        
        # Invalid performance weights
        with pytest.raises(ValueError, match="performance_weights must be a dictionary"):
            KMeansOptimizer(performance_weights="invalid")
    
    def test_prepare_features(self, sample_performance_data):
        """Test feature preparation for clustering."""
        optimizer = KMeansOptimizer()
        
        # Test with default weights
        features = optimizer._prepare_features(sample_performance_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_performance_data)
        assert 'composite_score' in features.columns
        assert features['composite_score'].dtype == np.float64
        
        # Check that composite score is properly calculated
        assert features['composite_score'].notna().all()
        assert features['composite_score'].var() > 0  # Should have variance
    
    def test_prepare_features_missing_columns(self):
        """Test feature preparation with missing columns."""
        optimizer = KMeansOptimizer()
        
        # Data missing required columns
        incomplete_data = pd.DataFrame({
            'sharpe_ratio': [1.0, 2.0, 1.5],
            'win_rate': [0.5, 0.6, 0.55]
            # Missing other required columns
        })
        
        with pytest.raises(ClusteringError, match="Missing required columns"):
            optimizer._prepare_features(incomplete_data)
    
    def test_prepare_features_custom_weights(self, sample_performance_data):
        """Test feature preparation with custom weights."""
        custom_weights = {
            'sharpe_ratio': 0.6,
            'max_drawdown': 0.4
        }
        
        optimizer = KMeansOptimizer(performance_weights=custom_weights)
        features = optimizer._prepare_features(sample_performance_data)
        
        assert isinstance(features, pd.DataFrame)
        assert 'composite_score' in features.columns
        
        # Verify only specified columns are used
        expected_columns = ['sharpe_ratio', 'max_drawdown', 'composite_score']
        assert all(col in features.columns for col in expected_columns)
    
    def test_normalize_features(self, sample_performance_data):
        """Test feature normalization."""
        optimizer = KMeansOptimizer()
        features = optimizer._prepare_features(sample_performance_data)
        
        normalized = optimizer._normalize_features(features)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape[0] == len(sample_performance_data)
        assert normalized.dtype == np.float64
        
        # Check normalization properties
        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-10)
    
    def test_find_optimal_clusters(self, sample_performance_data):
        """Test optimal cluster number finding."""
        optimizer = KMeansOptimizer(optimize_clusters=True)
        features = optimizer._prepare_features(sample_performance_data)
        normalized = optimizer._normalize_features(features)
        
        optimal_k = optimizer._find_optimal_clusters(normalized)
        
        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= 8  # Should be within reasonable range
    
    def test_find_optimal_clusters_insufficient_data(self):
        """Test optimal cluster finding with insufficient data."""
        optimizer = KMeansOptimizer(optimize_clusters=True)
        
        # Create very small dataset
        small_data = pd.DataFrame({
            'sharpe_ratio': [1.0, 2.0],
            'max_drawdown': [-0.1, -0.2],
            'win_rate': [0.5, 0.6],
            'profit_factor': [1.5, 2.0],
            'total_trades': [50, 60]
        })
        
        features = optimizer._prepare_features(small_data)
        normalized = optimizer._normalize_features(features)
        
        # Should fall back to default
        optimal_k = optimizer._find_optimal_clusters(normalized)
        assert optimal_k == 2  # Minimum clusters for small data
    
    def test_fit_basic(self, sample_performance_data):
        """Test basic clustering fit."""
        optimizer = KMeansOptimizer(n_clusters=3, optimize_clusters=False)
        
        result = optimizer.fit(sample_performance_data)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'cluster_centers' in result
        assert 'cluster_summary' in result
        assert 'best_cluster' in result
        assert 'silhouette_score' in result
        
        # Check cluster labels
        assert len(result['cluster_labels']) == len(sample_performance_data)
        assert all(0 <= label < 3 for label in result['cluster_labels'])
        
        # Check cluster centers
        assert len(result['cluster_centers']) == 3
        assert all(isinstance(center, dict) for center in result['cluster_centers'])
        
        # Check silhouette score
        assert -1 <= result['silhouette_score'] <= 1
    
    def test_fit_with_optimization(self, sample_performance_data):
        """Test clustering with cluster optimization."""
        optimizer = KMeansOptimizer(optimize_clusters=True)
        
        result = optimizer.fit(sample_performance_data)
        
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'optimal_clusters' in result
        assert 'cluster_centers' in result
        
        # Check that optimal clusters were found
        assert isinstance(result['optimal_clusters'], int)
        assert 2 <= result['optimal_clusters'] <= 8
        
        # Check that number of centers matches optimal clusters
        assert len(result['cluster_centers']) == result['optimal_clusters']
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        optimizer = KMeansOptimizer()
        
        with pytest.raises(ClusteringError, match="Data cannot be empty"):
            optimizer.fit(pd.DataFrame())
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        optimizer = KMeansOptimizer(n_clusters=5)
        
        # Create data with fewer samples than clusters
        small_data = pd.DataFrame({
            'sharpe_ratio': [1.0, 2.0, 1.5],
            'max_drawdown': [-0.1, -0.2, -0.15],
            'win_rate': [0.5, 0.6, 0.55],
            'profit_factor': [1.5, 2.0, 1.8],
            'total_trades': [50, 60, 55]
        })
        
        with pytest.raises(ClusteringError, match="Not enough data points"):
            optimizer.fit(small_data)
    
    def test_predict_new_data(self, sample_performance_data):
        """Test predicting clusters for new data."""
        optimizer = KMeansOptimizer(n_clusters=3)
        
        # Fit on training data
        optimizer.fit(sample_performance_data)
        
        # Create new data
        new_data = pd.DataFrame({
            'sharpe_ratio': [1.5, 0.8, 2.2],
            'max_drawdown': [-0.12, -0.20, -0.08],
            'win_rate': [0.60, 0.50, 0.65],
            'profit_factor': [2.0, 1.2, 2.8],
            'total_trades': [80, 60, 100]
        })
        
        predictions = optimizer.predict(new_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(new_data)
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_predict_not_fitted(self):
        """Test predicting without fitting first."""
        optimizer = KMeansOptimizer()
        
        new_data = pd.DataFrame({
            'sharpe_ratio': [1.5],
            'max_drawdown': [-0.12],
            'win_rate': [0.60],
            'profit_factor': [2.0],
            'total_trades': [80]
        })
        
        with pytest.raises(ClusteringError, match="Model has not been fitted"):
            optimizer.predict(new_data)
    
    def test_get_cluster_analysis(self, sample_performance_data):
        """Test cluster analysis functionality."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        analysis = optimizer.get_cluster_analysis(sample_performance_data, result['cluster_labels'])
        
        assert isinstance(analysis, dict)
        assert 'cluster_stats' in analysis
        assert 'cluster_rankings' in analysis
        assert 'best_performing_cluster' in analysis
        
        # Check cluster stats
        stats = analysis['cluster_stats']
        assert len(stats) == 3
        for cluster_id, cluster_stats in stats.items():
            assert 'size' in cluster_stats
            assert 'mean_performance' in cluster_stats
            assert 'std_performance' in cluster_stats
            assert isinstance(cluster_stats['size'], int)
            assert cluster_stats['size'] > 0
    
    def test_get_best_parameters(self, sample_performance_data):
        """Test getting best parameters from clustering."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        best_params = optimizer.get_best_parameters(sample_performance_data, result)
        
        assert isinstance(best_params, dict)
        assert 'cluster_id' in best_params
        assert 'parameters' in best_params
        assert 'expected_performance' in best_params
        
        # Check that parameters exist
        params = best_params['parameters']
        assert isinstance(params, dict)
        if 'atr_period' in sample_performance_data.columns:
            assert 'atr_period' in params
        if 'multiplier' in sample_performance_data.columns:
            assert 'multiplier' in params
    
    def test_cross_validation(self, sample_performance_data):
        """Test cross-validation functionality."""
        optimizer = KMeansOptimizer(n_clusters=3)
        
        cv_scores = optimizer.cross_validate(sample_performance_data, cv_folds=3)
        
        assert isinstance(cv_scores, dict)
        assert 'silhouette_scores' in cv_scores
        assert 'mean_silhouette' in cv_scores
        assert 'std_silhouette' in cv_scores
        assert 'stability_score' in cv_scores
        
        # Check scores
        scores = cv_scores['silhouette_scores']
        assert len(scores) == 3
        assert all(-1 <= score <= 1 for score in scores)
        
        # Check aggregated metrics
        assert -1 <= cv_scores['mean_silhouette'] <= 1
        assert cv_scores['std_silhouette'] >= 0
        assert 0 <= cv_scores['stability_score'] <= 1
    
    def test_hyperparameter_tuning(self, sample_performance_data):
        """Test hyperparameter tuning."""
        optimizer = KMeansOptimizer()
        
        param_grid = {
            'n_clusters': [2, 3, 4],
            'max_iter': [100, 300]
        }
        
        best_params = optimizer.tune_hyperparameters(sample_performance_data, param_grid)
        
        assert isinstance(best_params, dict)
        assert 'best_params' in best_params
        assert 'best_score' in best_params
        assert 'all_results' in best_params
        
        # Check best parameters
        best = best_params['best_params']
        assert 'n_clusters' in best
        assert 'max_iter' in best
        assert best['n_clusters'] in [2, 3, 4]
        assert best['max_iter'] in [100, 300]
        
        # Check score
        assert -1 <= best_params['best_score'] <= 1
    
    def test_cluster_stability(self, sample_performance_data):
        """Test cluster stability analysis."""
        optimizer = KMeansOptimizer(n_clusters=3)
        
        stability = optimizer.analyze_cluster_stability(sample_performance_data, n_runs=5)
        
        assert isinstance(stability, dict)
        assert 'stability_scores' in stability
        assert 'mean_stability' in stability
        assert 'consensus_labels' in stability
        
        # Check stability scores
        scores = stability['stability_scores']
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)
        
        # Check consensus labels
        consensus = stability['consensus_labels']
        assert len(consensus) == len(sample_performance_data)
        assert all(0 <= label < 3 for label in consensus)
    
    def test_feature_importance(self, sample_performance_data):
        """Test feature importance analysis."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        importance = optimizer.analyze_feature_importance(sample_performance_data, result)
        
        assert isinstance(importance, dict)
        assert 'feature_weights' in importance
        assert 'separation_scores' in importance
        assert 'cluster_contributions' in importance
        
        # Check feature weights
        weights = importance['feature_weights']
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(weight >= 0 for weight in weights.values())
        
        # Check separation scores
        sep_scores = importance['separation_scores']
        assert isinstance(sep_scores, dict)
        assert len(sep_scores) > 0
    
    def test_outlier_detection(self, sample_performance_data):
        """Test outlier detection in clustering."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        outliers = optimizer.detect_outliers(sample_performance_data, result, threshold=2.0)
        
        assert isinstance(outliers, dict)
        assert 'outlier_indices' in outliers
        assert 'outlier_scores' in outliers
        assert 'threshold' in outliers
        
        # Check outlier indices
        indices = outliers['outlier_indices']
        assert isinstance(indices, (list, np.ndarray))
        assert all(0 <= idx < len(sample_performance_data) for idx in indices)
        
        # Check outlier scores
        scores = outliers['outlier_scores']
        assert len(scores) == len(sample_performance_data)
        assert all(score >= 0 for score in scores)
    
    def test_cluster_visualization_data(self, sample_performance_data):
        """Test cluster visualization data preparation."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        viz_data = optimizer.prepare_visualization_data(sample_performance_data, result)
        
        assert isinstance(viz_data, dict)
        assert 'points' in viz_data
        assert 'centroids' in viz_data
        assert 'cluster_labels' in viz_data
        assert 'explained_variance' in viz_data
        
        # Check points data
        points = viz_data['points']
        assert isinstance(points, np.ndarray)
        assert points.shape[0] == len(sample_performance_data)
        assert points.shape[1] == 2  # 2D projection
        
        # Check centroids
        centroids = viz_data['centroids']
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[0] == 3  # 3 clusters
        assert centroids.shape[1] == 2  # 2D projection
    
    def test_save_load_model(self, sample_performance_data, tmp_path):
        """Test model saving and loading."""
        optimizer = KMeansOptimizer(n_clusters=3)
        optimizer.fit(sample_performance_data)
        
        # Save model
        model_path = tmp_path / "kmeans_model.pkl"
        optimizer.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_optimizer = KMeansOptimizer()
        new_optimizer.load_model(str(model_path))
        
        # Test that loaded model works
        new_data = sample_performance_data.head(10)
        predictions = new_optimizer.predict(new_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
    
    def test_model_serialization(self, sample_performance_data):
        """Test model serialization to dictionary."""
        optimizer = KMeansOptimizer(n_clusters=3)
        result = optimizer.fit(sample_performance_data)
        
        model_dict = optimizer.to_dict()
        
        assert isinstance(model_dict, dict)
        assert 'model_type' in model_dict
        assert 'parameters' in model_dict
        assert 'cluster_centers' in model_dict
        assert 'scaler_params' in model_dict
        
        # Test deserialization
        new_optimizer = KMeansOptimizer.from_dict(model_dict)
        
        assert new_optimizer.n_clusters == optimizer.n_clusters
        assert new_optimizer.random_state == optimizer.random_state
    
    def test_integration_with_supertrend(self, sample_performance_data):
        """Test integration with SuperTrend parameter optimization."""
        optimizer = KMeansOptimizer(n_clusters=3)
        
        # Add SuperTrend specific parameters
        sample_performance_data['atr_period'] = np.random.choice([7, 10, 14, 20], len(sample_performance_data))
        sample_performance_data['multiplier'] = np.random.choice([1.5, 2.0, 2.5, 3.0], len(sample_performance_data))
        
        result = optimizer.fit(sample_performance_data)
        
        # Get best SuperTrend parameters
        best_params = optimizer.get_best_parameters(sample_performance_data, result)
        
        assert 'atr_period' in best_params['parameters']
        assert 'multiplier' in best_params['parameters']
        assert best_params['parameters']['atr_period'] in [7, 10, 14, 20]
        assert best_params['parameters']['multiplier'] in [1.5, 2.0, 2.5, 3.0]
    
    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'sharpe_ratio': np.random.normal(1.0, 0.5, 10000),
            'max_drawdown': np.random.normal(-0.15, 0.05, 10000),
            'win_rate': np.random.normal(0.55, 0.1, 10000),
            'profit_factor': np.random.normal(1.5, 0.5, 10000),
            'total_trades': np.random.normal(80, 20, 10000)
        })
        
        optimizer = KMeansOptimizer(n_clusters=5)
        
        import time
        start_time = time.time()
        result = optimizer.fit(large_data)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30  # Less than 30 seconds
        
        # Should produce valid results
        assert isinstance(result, dict)
        assert len(result['cluster_labels']) == 10000
        assert len(result['cluster_centers']) == 5


class TestClusteringIntegration:
    """Integration tests for clustering functionality."""
    
    def test_clustering_with_backtest_results(self):
        """Test clustering with actual backtest results."""
        # Create realistic backtest results
        np.random.seed(42)
        n_runs = 200
        
        backtest_results = pd.DataFrame({
            'strategy_id': range(n_runs),
            'atr_period': np.random.choice([7, 10, 14, 20], n_runs),
            'multiplier': np.random.choice([1.5, 2.0, 2.5, 3.0], n_runs),
            'sharpe_ratio': np.random.normal(1.0, 0.8, n_runs),
            'max_drawdown': np.random.normal(-0.15, 0.08, n_runs),
            'win_rate': np.random.normal(0.55, 0.15, n_runs),
            'profit_factor': np.random.normal(1.5, 0.8, n_runs),
            'total_trades': np.random.normal(80, 30, n_runs),
            'annual_return': np.random.normal(0.12, 0.15, n_runs),
            'calmar_ratio': np.random.normal(0.8, 0.5, n_runs)
        })
        
        # Ensure non-negative values where appropriate
        backtest_results['win_rate'] = backtest_results['win_rate'].clip(0, 1)
        backtest_results['total_trades'] = backtest_results['total_trades'].clip(1, None)
        
        optimizer = KMeansOptimizer(
            n_clusters=3,
            optimize_clusters=True,
            performance_weights={
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.2,
                'win_rate': 0.2,
                'profit_factor': 0.15,
                'total_trades': 0.1,
                'annual_return': 0.05
            }
        )
        
        result = optimizer.fit(backtest_results)
        
        # Verify clustering worked
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'best_cluster' in result
        
        # Get best performing cluster
        best_cluster_id = result['best_cluster']
        best_cluster_mask = result['cluster_labels'] == best_cluster_id
        best_cluster_data = backtest_results[best_cluster_mask]
        
        # Best cluster should have better average performance
        assert best_cluster_data['sharpe_ratio'].mean() > backtest_results['sharpe_ratio'].mean()
        assert best_cluster_data['max_drawdown'].mean() > backtest_results['max_drawdown'].mean()
        assert best_cluster_data['win_rate'].mean() > backtest_results['win_rate'].mean()
    
    def test_clustering_pipeline_robustness(self):
        """Test clustering pipeline robustness with various data conditions."""
        optimizer = KMeansOptimizer(n_clusters=3)
        
        # Test with missing values
        data_with_nans = pd.DataFrame({
            'sharpe_ratio': [1.0, np.nan, 2.0, 1.5, np.nan],
            'max_drawdown': [-0.1, -0.2, np.nan, -0.15, -0.12],
            'win_rate': [0.5, 0.6, 0.55, np.nan, 0.58],
            'profit_factor': [1.5, 2.0, 1.8, 1.6, np.nan],
            'total_trades': [50, 60, 55, 58, 52]
        })
        
        # Should handle NaN values gracefully
        with pytest.raises(ClusteringError):
            optimizer.fit(data_with_nans)
        
        # Test with all identical values
        identical_data = pd.DataFrame({
            'sharpe_ratio': [1.0] * 10,
            'max_drawdown': [-0.1] * 10,
            'win_rate': [0.5] * 10,
            'profit_factor': [1.5] * 10,
            'total_trades': [50] * 10
        })
        
        # Should handle identical values
        with pytest.raises(ClusteringError):
            optimizer.fit(identical_data)
    
    @patch('src.ml.clustering.kmeans_optimizer.KMeans')
    def test_clustering_with_sklearn_errors(self, mock_kmeans):
        """Test clustering behavior when sklearn raises errors."""
        # Mock sklearn to raise an error
        mock_kmeans.side_effect = Exception("sklearn error")
        
        optimizer = KMeansOptimizer(n_clusters=3)
        
        data = pd.DataFrame({
            'sharpe_ratio': [1.0, 2.0, 1.5],
            'max_drawdown': [-0.1, -0.2, -0.15],
            'win_rate': [0.5, 0.6, 0.55],
            'profit_factor': [1.5, 2.0, 1.8],
            'total_trades': [50, 60, 55]
        })
        
        with pytest.raises(ClusteringError):
            optimizer.fit(data)
    
    def test_clustering_memory_efficiency(self):
        """Test clustering memory efficiency with large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'sharpe_ratio': np.random.normal(1.0, 0.5, 50000),
            'max_drawdown': np.random.normal(-0.15, 0.05, 50000),
            'win_rate': np.random.normal(0.55, 0.1, 50000),
            'profit_factor': np.random.normal(1.5, 0.5, 50000),
            'total_trades': np.random.normal(80, 20, 50000)
        })
        
        optimizer = KMeansOptimizer(n_clusters=5)
        
        # Monitor memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = optimizer.fit(large_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_increase < 500  # Less than 500MB increase
        
        # Should still produce valid results
        assert isinstance(result, dict)
        assert len(result['cluster_labels']) == 50000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])