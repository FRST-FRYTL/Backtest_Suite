"""
Feature Selection Module for ML Trading System
Implements various feature selection techniques to identify the most relevant features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection class that implements multiple selection methods
    to identify the most important features for trading predictions
    """
    
    def __init__(self, task_type: str = 'regression', n_jobs: int = -1):
        """
        Initialize feature selector
        
        Args:
            task_type: 'regression' or 'classification'
            n_jobs: Number of parallel jobs
        """
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.feature_importance_ = {}
        self.selected_features_ = []
        self.selection_methods_ = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 100,
                       methods: List[str] = ['mutual_info', 'random_forest', 'lasso', 'correlation']) -> pd.DataFrame:
        """
        Main method to select top features using multiple methods
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            methods: List of selection methods to use
            
        Returns:
            DataFrame with selected features
        """
        # Handle missing values
        X_filled = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Apply different selection methods
        all_scores = {}
        
        if 'mutual_info' in methods:
            mi_scores = self._mutual_information_selection(X_filled, y)
            all_scores['mutual_info'] = mi_scores
            
        if 'random_forest' in methods:
            rf_scores = self._random_forest_importance(X_filled, y)
            all_scores['random_forest'] = rf_scores
            
        if 'lasso' in methods:
            lasso_scores = self._lasso_selection(X_filled, y)
            all_scores['lasso'] = lasso_scores
            
        if 'correlation' in methods:
            corr_scores = self._correlation_selection(X_filled, y)
            all_scores['correlation'] = corr_scores
            
        if 'multicollinearity' in methods:
            multicol_features = self._remove_multicollinearity(X_filled)
            all_scores['multicollinearity'] = multicol_features
            
        # Combine scores using rank aggregation
        combined_scores = self._aggregate_scores(all_scores)
        
        # Select top features
        top_features = combined_scores.nlargest(n_features).index.tolist()
        self.selected_features_ = top_features
        
        # Store feature importance
        self.feature_importance_ = combined_scores.to_dict()
        
        return X[top_features]
    
    def _mutual_information_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Select features using mutual information"""
        if self.task_type == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
            
        mi_scores = pd.Series(mi_scores, index=X.columns)
        self.selection_methods_['mutual_info'] = mi_scores
        
        return mi_scores
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Select features using Random Forest importance"""
        if self.task_type == 'regression':
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=self.n_jobs
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=self.n_jobs
            )
            
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        self.selection_methods_['random_forest'] = importances
        
        return importances
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Select features using Lasso regularization"""
        # Scale features for Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use LassoCV to find optimal alpha
        lasso = LassoCV(cv=5, random_state=42, n_jobs=self.n_jobs)
        lasso.fit(X_scaled, y)
        
        # Get absolute coefficients as importance scores
        coef_abs = np.abs(lasso.coef_)
        lasso_scores = pd.Series(coef_abs, index=X.columns)
        
        self.selection_methods_['lasso'] = lasso_scores
        
        return lasso_scores
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Select features based on correlation with target"""
        correlations = X.corrwith(y).abs()
        self.selection_methods_['correlation'] = correlations
        
        return correlations
    
    def _remove_multicollinearity(self, X: pd.DataFrame, 
                                 threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features to reduce multicollinearity
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to keep
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Keep features that are not highly correlated
        features_to_keep = [col for col in X.columns if col not in to_drop]
        
        return features_to_keep
    
    def _aggregate_scores(self, scores_dict: Dict[str, pd.Series]) -> pd.Series:
        """
        Aggregate scores from different methods using rank aggregation
        
        Args:
            scores_dict: Dictionary of scores from different methods
            
        Returns:
            Aggregated scores
        """
        # Convert scores to ranks
        ranks_dict = {}
        for method, scores in scores_dict.items():
            if isinstance(scores, list):
                # For multicollinearity, create binary scores
                all_features = set()
                for s in scores_dict.values():
                    if isinstance(s, pd.Series):
                        all_features.update(s.index)
                binary_scores = pd.Series(
                    [1 if feat in scores else 0 for feat in all_features],
                    index=list(all_features)
                )
                ranks_dict[method] = binary_scores.rank(ascending=False)
            else:
                # Rank features (higher score = better rank)
                ranks_dict[method] = scores.rank(ascending=False)
        
        # Calculate average rank
        ranks_df = pd.DataFrame(ranks_dict)
        avg_ranks = ranks_df.mean(axis=1)
        
        # Convert average ranks back to scores (lower rank = higher score)
        aggregated_scores = 1 / avg_ranks
        
        return aggregated_scores
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    n_features: int = 50,
                                    cv: int = 5) -> List[str]:
        """
        Perform recursive feature elimination with cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            cv: Number of cross-validation folds
            
        Returns:
            List of selected features
        """
        # Handle missing values
        X_filled = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create base estimator
        if self.task_type == 'regression':
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=self.n_jobs
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=self.n_jobs
            )
        
        # Perform RFECV
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy',
            n_jobs=self.n_jobs
        )
        
        selector.fit(X_filled, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        
        return selected_features[:n_features]
    
    def variance_threshold_selection(self, X: pd.DataFrame, 
                                   threshold: float = 0.01) -> List[str]:
        """
        Remove features with low variance
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            List of features with sufficient variance
        """
        # Calculate variance
        variances = X.var()
        
        # Select features above threshold
        selected_features = variances[variances > threshold].index.tolist()
        
        return selected_features
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate comprehensive feature importance report
        
        Returns:
            DataFrame with feature importance from all methods
        """
        # Combine all selection methods results
        report_data = {}
        
        # Add individual method scores
        for method, scores in self.selection_methods_.items():
            if isinstance(scores, pd.Series):
                report_data[f'{method}_score'] = scores
                report_data[f'{method}_rank'] = scores.rank(ascending=False)
        
        # Add final importance scores
        if self.feature_importance_:
            report_data['final_score'] = pd.Series(self.feature_importance_)
            report_data['selected'] = pd.Series(
                {feat: feat in self.selected_features_ 
                 for feat in report_data['final_score'].index}
            )
        
        # Create report DataFrame
        report = pd.DataFrame(report_data)
        
        # Sort by final score
        if 'final_score' in report.columns:
            report = report.sort_values('final_score', ascending=False)
        
        return report
    
    def plot_feature_importance(self, top_n: int = 30):
        """
        Plot feature importance scores
        
        Args:
            top_n: Number of top features to plot
        """
        import matplotlib.pyplot as plt
        
        if not self.feature_importance_:
            raise ValueError("No feature importance scores available. Run select_features first.")
        
        # Get top features
        importance_series = pd.Series(self.feature_importance_).sort_values(ascending=True)
        top_features = importance_series.tail(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        y_positions = np.arange(len(top_features))
        ax.barh(y_positions, top_features.values)
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance Scores')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def get_correlation_analysis(self, X: pd.DataFrame, 
                               selected_features: Optional[List[str]] = None) -> Dict:
        """
        Analyze correlations between selected features
        
        Args:
            X: Feature DataFrame
            selected_features: List of features to analyze (uses self.selected_features_ if None)
            
        Returns:
            Dictionary with correlation analysis results
        """
        if selected_features is None:
            selected_features = self.selected_features_
            
        if not selected_features:
            raise ValueError("No selected features available")
            
        # Calculate correlation matrix for selected features
        X_selected = X[selected_features]
        corr_matrix = X_selected.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Calculate average absolute correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = np.abs(corr_matrix.where(mask)).mean().mean()
        
        analysis = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': pd.DataFrame(high_corr_pairs),
            'average_absolute_correlation': avg_correlation,
            'max_correlation': corr_matrix.where(mask).max().max(),
            'min_correlation': corr_matrix.where(mask).min().min()
        }
        
        return analysis
    
    def stability_analysis(self, X: pd.DataFrame, y: pd.Series,
                          n_iterations: int = 10,
                          subsample_ratio: float = 0.8) -> pd.DataFrame:
        """
        Analyze feature selection stability across different data subsamples
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_iterations: Number of stability iterations
            subsample_ratio: Ratio of data to use in each iteration
            
        Returns:
            DataFrame with stability scores for each feature
        """
        feature_counts = {}
        
        for i in range(n_iterations):
            # Subsample data
            n_samples = int(len(X) * subsample_ratio)
            indices = np.random.choice(X.index, size=n_samples, replace=False)
            X_sub = X.loc[indices]
            y_sub = y.loc[indices]
            
            # Select features
            selected = self.select_features(X_sub, y_sub, n_features=100, 
                                          methods=['mutual_info', 'random_forest'])
            
            # Count selected features
            for feat in selected.columns:
                if feat not in feature_counts:
                    feature_counts[feat] = 0
                feature_counts[feat] += 1
        
        # Calculate stability scores
        stability_scores = pd.Series(feature_counts) / n_iterations
        stability_df = pd.DataFrame({
            'stability_score': stability_scores,
            'selection_frequency': pd.Series(feature_counts)
        })
        
        return stability_df.sort_values('stability_score', ascending=False)