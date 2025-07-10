"""
Feature Analysis Agent for ML Pipeline

Analyzes feature importance, relationships, and quality for optimal
feature selection in the machine learning pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    SelectKBest, f_classif, f_regression,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class FeatureAnalysisAgent(BaseAgent):
    """
    Agent responsible for feature analysis including:
    - Feature importance calculation
    - Correlation analysis
    - Mutual information scoring
    - Feature selection
    - Dimensionality reduction
    - Feature interaction analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FeatureAnalysisAgent", config)
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        self.selected_features = []
        self.feature_relationships = {}
        self.pca_components = None
        
    def initialize(self) -> bool:
        """Initialize feature analysis resources."""
        try:
            self.logger.info("Initializing Feature Analysis Agent")
            
            # Validate required configuration
            required_keys = ["analysis_methods", "selection_criteria", "target_variable"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize analysis methods
            self.analysis_methods = self.config.get("analysis_methods", [
                "correlation", "mutual_information", "random_forest", "permutation"
            ])
            
            # Initialize selection criteria
            self.selection_criteria = self.config.get("selection_criteria", {
                "max_features": 50,
                "min_importance": 0.01,
                "correlation_threshold": 0.95
            })
            
            # Set target variable
            self.target_variable = self.config.get("target_variable", "returns")
            
            # Initialize task type (classification or regression)
            self.task_type = self.config.get("task_type", "regression")
            
            self.logger.info("Feature Analysis Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, data: pd.DataFrame, target: Optional[pd.Series] = None, 
                **kwargs) -> Dict[str, Any]:
        """
        Execute feature analysis pipeline.
        
        Args:
            data: Feature dataframe
            target: Target variable series
            
        Returns:
            Dict containing analysis results and selected features
        """
        try:
            # Prepare target variable
            if target is None:
                if self.target_variable in data.columns:
                    target = data[self.target_variable]
                    features = data.drop(columns=[self.target_variable])
                else:
                    # Generate synthetic target for demonstration
                    target = self._generate_synthetic_target(data)
                    features = data
            else:
                features = data
            
            # Remove non-numeric columns
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Correlation analysis
            correlation_results = self._analyze_correlations(numeric_features, target)
            
            # Mutual information analysis
            mi_scores = self._calculate_mutual_information(numeric_features, target)
            
            # Random forest importance
            rf_importance = self._calculate_rf_importance(numeric_features, target)
            
            # Permutation importance
            perm_importance = self._calculate_permutation_importance(
                numeric_features, target
            )
            
            # Combine importance scores
            self._combine_importance_scores({
                "correlation": correlation_results["feature_target_corr"],
                "mutual_info": mi_scores,
                "random_forest": rf_importance,
                "permutation": perm_importance
            })
            
            # Select features
            selected_features = self._select_features(numeric_features)
            
            # Analyze feature interactions
            interaction_results = self._analyze_feature_interactions(
                numeric_features[selected_features], target
            )
            
            # Dimensionality reduction analysis
            pca_results = self._perform_pca_analysis(numeric_features[selected_features])
            
            # Generate visualizations
            viz_results = self._generate_visualizations(
                numeric_features[selected_features], target
            )
            
            return {
                "selected_features": selected_features,
                "feature_importance": self.feature_importance_scores,
                "correlation_analysis": correlation_results,
                "interaction_analysis": interaction_results,
                "pca_analysis": pca_results,
                "feature_count": {
                    "original": len(numeric_features.columns),
                    "selected": len(selected_features)
                },
                "visualizations": viz_results
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _generate_synthetic_target(self, data: pd.DataFrame) -> pd.Series:
        """Generate synthetic target for demonstration."""
        # Simple momentum-based target
        if 'returns' in data.columns:
            return (data['returns'].shift(-1) > 0).astype(int)
        else:
            # Random target
            return pd.Series(
                np.random.randint(0, 2, len(data)), 
                index=data.index,
                name='target'
            )
    
    def _analyze_correlations(self, features: pd.DataFrame, 
                            target: pd.Series) -> Dict[str, Any]:
        """Analyze feature correlations."""
        self.logger.info("Analyzing feature correlations")
        
        # Feature-feature correlations
        self.correlation_matrix = features.corr()
        
        # Feature-target correlations
        feature_target_corr = {}
        for col in features.columns:
            corr, p_value = pearsonr(features[col].dropna(), 
                                    target.loc[features[col].dropna().index])
            feature_target_corr[col] = {
                "correlation": corr,
                "p_value": p_value,
                "abs_correlation": abs(corr)
            }
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        threshold = self.selection_criteria.get("correlation_threshold", 0.95)
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                if abs(self.correlation_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        "feature1": self.correlation_matrix.columns[i],
                        "feature2": self.correlation_matrix.columns[j],
                        "correlation": self.correlation_matrix.iloc[i, j]
                    })
        
        return {
            "feature_target_corr": feature_target_corr,
            "high_correlation_pairs": high_corr_pairs,
            "correlation_matrix": self.correlation_matrix
        }
    
    def _calculate_mutual_information(self, features: pd.DataFrame, 
                                    target: pd.Series) -> Dict[str, float]:
        """Calculate mutual information scores."""
        self.logger.info("Calculating mutual information scores")
        
        # Handle missing values
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[mask]
        target_clean = target[mask]
        
        if self.task_type == "classification":
            mi_scores = mutual_info_classif(features_clean, target_clean)
        else:
            mi_scores = mutual_info_regression(features_clean, target_clean)
        
        return {col: score for col, score in zip(features.columns, mi_scores)}
    
    def _calculate_rf_importance(self, features: pd.DataFrame, 
                               target: pd.Series) -> Dict[str, float]:
        """Calculate Random Forest feature importance."""
        self.logger.info("Calculating Random Forest importance")
        
        # Handle missing values
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[mask]
        target_clean = target[mask]
        
        if self.task_type == "classification":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(features_clean, target_clean)
        
        return {col: imp for col, imp in 
                zip(features.columns, rf.feature_importances_)}
    
    def _calculate_permutation_importance(self, features: pd.DataFrame, 
                                        target: pd.Series) -> Dict[str, float]:
        """Calculate permutation importance."""
        self.logger.info("Calculating permutation importance")
        
        # Handle missing values
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[mask]
        target_clean = target[mask]
        
        # Use a simple model for permutation importance
        if self.task_type == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        model.fit(features_clean, target_clean)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, features_clean, target_clean, 
            n_repeats=10, random_state=42
        )
        
        return {col: imp for col, imp in 
                zip(features.columns, perm_importance.importances_mean)}
    
    def _combine_importance_scores(self, scores_dict: Dict[str, Dict[str, float]]):
        """Combine importance scores from different methods."""
        self.logger.info("Combining importance scores")
        
        all_features = set()
        for scores in scores_dict.values():
            all_features.update(scores.keys())
        
        # Normalize scores for each method
        normalized_scores = {}
        for method, scores in scores_dict.items():
            if not scores:
                continue
            max_score = max(abs(v) if isinstance(v, (int, float)) else 
                          abs(v["abs_correlation"]) if isinstance(v, dict) else 0
                          for v in scores.values())
            if max_score > 0:
                normalized_scores[method] = {
                    feature: (abs(score) if isinstance(score, (int, float)) else
                             abs(score["abs_correlation"]) if isinstance(score, dict) else 0) / max_score
                    for feature, score in scores.items()
                }
        
        # Combine scores with equal weighting
        self.feature_importance_scores = {}
        for feature in all_features:
            scores = []
            for method, method_scores in normalized_scores.items():
                if feature in method_scores:
                    scores.append(method_scores[feature])
            
            if scores:
                self.feature_importance_scores[feature] = {
                    "combined_score": np.mean(scores),
                    "individual_scores": {
                        method: normalized_scores[method].get(feature, 0)
                        for method in normalized_scores
                    }
                }
    
    def _select_features(self, features: pd.DataFrame) -> List[str]:
        """Select features based on importance scores and criteria."""
        self.logger.info("Selecting features")
        
        # Sort features by combined importance score
        sorted_features = sorted(
            self.feature_importance_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )
        
        # Apply selection criteria
        selected = []
        min_importance = self.selection_criteria.get("min_importance", 0.01)
        max_features = self.selection_criteria.get("max_features", 50)
        
        for feature, scores in sorted_features:
            if len(selected) >= max_features:
                break
            
            if scores["combined_score"] >= min_importance:
                # Check if feature is highly correlated with already selected features
                is_redundant = False
                for selected_feature in selected:
                    if (feature in self.correlation_matrix.columns and 
                        selected_feature in self.correlation_matrix.columns):
                        corr = abs(self.correlation_matrix.loc[feature, selected_feature])
                        if corr > self.selection_criteria.get("correlation_threshold", 0.95):
                            is_redundant = True
                            break
                
                if not is_redundant:
                    selected.append(feature)
        
        self.selected_features = selected
        self.logger.info(f"Selected {len(selected)} features")
        
        return selected
    
    def _analyze_feature_interactions(self, features: pd.DataFrame, 
                                    target: pd.Series) -> Dict[str, Any]:
        """Analyze feature interactions."""
        self.logger.info("Analyzing feature interactions")
        
        interactions = {}
        
        # Analyze top feature pairs
        top_features = self.selected_features[:10]  # Limit to top 10 for efficiency
        
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if feat1 in features.columns and feat2 in features.columns:
                    # Create interaction feature
                    interaction = features[feat1] * features[feat2]
                    
                    # Calculate correlation with target
                    corr, p_value = pearsonr(
                        interaction.dropna(),
                        target.loc[interaction.dropna().index]
                    )
                    
                    interactions[f"{feat1}_x_{feat2}"] = {
                        "correlation": corr,
                        "p_value": p_value,
                        "stronger_than_individual": (
                            abs(corr) > abs(self.feature_importance_scores[feat1]["combined_score"]) and
                            abs(corr) > abs(self.feature_importance_scores[feat2]["combined_score"])
                        )
                    }
        
        return {
            "interaction_count": len(interactions),
            "strong_interactions": [
                k for k, v in interactions.items() 
                if v["stronger_than_individual"]
            ],
            "all_interactions": interactions
        }
    
    def _perform_pca_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA analysis for dimensionality reduction."""
        self.logger.info("Performing PCA analysis")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.dropna())
        
        # Perform PCA
        pca = PCA()
        pca_features = pca.fit_transform(features_scaled)
        
        # Calculate cumulative explained variance
        cumvar = pca.explained_variance_ratio_.cumsum()
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumvar >= 0.95) + 1
        
        self.pca_components = pca
        
        return {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumvar.tolist(),
            "n_components_95": int(n_components_95),
            "total_components": len(pca.explained_variance_ratio_),
            "dimension_reduction": f"{n_components_95}/{len(features.columns)} "
                                 f"({n_components_95/len(features.columns)*100:.1f}%)"
        }
    
    def _generate_visualizations(self, features: pd.DataFrame, 
                               target: pd.Series) -> Dict[str, str]:
        """Generate feature analysis visualizations."""
        self.logger.info("Generating visualizations")
        
        viz_paths = {}
        
        try:
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            top_features = sorted(
                self.feature_importance_scores.items(),
                key=lambda x: x[1]["combined_score"],
                reverse=True
            )[:20]
            
            features_names = [f[0] for f in top_features]
            importance_values = [f[1]["combined_score"] for f in top_features]
            
            plt.barh(features_names, importance_values)
            plt.xlabel('Importance Score')
            plt.title('Top 20 Feature Importance Scores')
            plt.tight_layout()
            plt.savefig('/tmp/feature_importance.png')
            viz_paths["feature_importance"] = '/tmp/feature_importance.png'
            plt.close()
            
            # Correlation heatmap
            if len(self.selected_features) <= 30:
                plt.figure(figsize=(12, 10))
                selected_corr = self.correlation_matrix.loc[
                    self.selected_features, self.selected_features
                ]
                sns.heatmap(selected_corr, cmap='coolwarm', center=0, 
                           annot=False, square=True)
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('/tmp/correlation_heatmap.png')
                viz_paths["correlation_heatmap"] = '/tmp/correlation_heatmap.png'
                plt.close()
            
            # PCA variance plot
            if self.pca_components is not None:
                plt.figure(figsize=(10, 6))
                cumvar = self.pca_components.explained_variance_ratio_.cumsum()
                plt.plot(range(1, len(cumvar) + 1), cumvar, 'b-', marker='o')
                plt.axhline(y=0.95, color='r', linestyle='--', 
                           label='95% variance threshold')
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')
                plt.title('PCA Explained Variance')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('/tmp/pca_variance.png')
                viz_paths["pca_variance"] = '/tmp/pca_variance.png'
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {str(e)}")
        
        return viz_paths
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature analysis report."""
        return {
            "selected_features": self.selected_features,
            "feature_scores": self.feature_importance_scores,
            "feature_count": len(self.selected_features),
            "reduction_ratio": len(self.selected_features) / len(self.feature_importance_scores)
            if self.feature_importance_scores else 0
        }