# src/analysis/reporting.py - Generates analysis reports for data and models

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import os
import logging
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

# Optional SciPy import for Q-Q plots
try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:
    _scipy_stats = None

# Make SHAP optional to avoid build tool requirements on Windows
try:
    import shap  # type: ignore
except Exception:
    shap = None

class Reporting:
    """Handles all reporting and analysis for the ML pipeline using ReportLab."""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.report_dir = self.config.get('reports.directory', 'reports')
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_full_report(self, df: pd.DataFrame, feature_lineage: list, models_data: dict, X: pd.DataFrame, y: pd.Series, cleaned_stats: dict, recommendations: dict = None):
        """
        Generate a comprehensive PDF report with all analyses.
        
        Args:
            df: DataFrame containing the processed data
            feature_lineage: List of feature lineage information
            models_data: Dictionary containing trained models and their metrics
            X: Feature matrix
            y: Target variable
            cleaned_stats: Dictionary with data cleaning statistics
            recommendations: Dictionary containing optimization results and recommendations
        """
        try:
            self.logger.info("Starting report generation...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(self.report_dir, f'fantasy_football_report_{timestamp}.pdf')
            doc = SimpleDocTemplate(pdf_path, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            styles = getSampleStyleSheet()
            story = []

            # Add title and timestamp
            story.append(Paragraph("Fantasy Football ML Pipeline Report", styles['h1']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
            story.append(Spacer(1, 0.5*inch))

            # 1. Executive Summary
            story.append(Paragraph("Executive Summary", styles['h2']))
            story.append(Paragraph(
                "This report provides a comprehensive analysis of the Fantasy Premier League (FPL) optimization pipeline, "
                "including data processing, feature engineering, model performance, and team optimization results.",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.3*inch))

            # 2. Data Overview
            story.append(Paragraph("Data Overview", styles['h2']))
            story.append(Paragraph(
                f"The analysis includes data for {len(df)} players with {len(feature_lineage)} engineered features. "
                f"The dataset has been cleaned and processed for optimal model performance.",
                styles['Normal']
            ))
            
            # Add data summary table
            if not df.empty:
                summary_data = [
                    ['Metric', 'Value'],
                    ['Total Players', len(df)],
                    ['Total Features', len(feature_lineage) if feature_lineage else 'N/A'],
                    ['Target Variable', 'total_points'],
                    ['Data Period', f"GW{df['round'].min()} to GW{df['round'].max()}" if 'round' in df.columns else 'N/A']
                ]
                
                table = Table(summary_data, colWidths=[2*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),  # Blue header
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D9E1F2')),  # Light blue rows
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#8EA9DB')),    # Lighter blue grid
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                ]))
                story.append(table)
            
            story.append(PageBreak())

            # 3. Data Cleaning Report
            try:
                story.append(Paragraph("Data Cleaning Report", styles['h2']))
                cleaning_report = self._generate_cleaning_report(cleaned_stats, styles)
                if cleaning_report:
                    story.extend(cleaning_report)
                else:
                    story.append(Paragraph("No data cleaning statistics available.", styles['Normal']))
                story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating cleaning report: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating data cleaning report.", styles['Normal']))
                story.append(PageBreak())

            # 4. Feature Analysis
            try:
                story.append(Paragraph("Feature Analysis", styles['h2']))
                story.append(Paragraph(
                    "This section provides an overview of the input features used in the machine learning pipeline. "
                    "It includes summary statistics and visualizations of the feature distributions, as well as information on feature lineage.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 12))
                
                feature_analysis = self._generate_feature_analysis(df, feature_lineage, styles)
                if feature_analysis:
                    story.extend(feature_analysis)
                else:
                    story.append(Paragraph("No feature analysis available.", styles['Normal']))
                story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating feature analysis: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating feature analysis.", styles['Normal']))
                story.append(PageBreak())

            # 5. Feature Importance Analysis
            try:
                story.append(Paragraph("Feature Importance Analysis", styles['h2']))
                story.append(Paragraph(
                    "This section presents the results of the Boruta feature importance analysis, which identifies "
                    "the most informative features for predicting player points. Features in green are confirmed to be important, "
                    "yellow features are tentatively important, and red features are confirmed to be unimportant.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 12))
                
                boruta_report = self._generate_boruta_report(X, y, styles)
                if boruta_report:
                    story.extend(boruta_report)
                else:
                    story.append(Paragraph("No feature importance analysis available.", styles['Normal']))
                story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating feature importance analysis: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating feature importance analysis.", styles['Normal']))
                story.append(PageBreak())

            # 5. Model Diagnostics
            try:
                if models_data:
                    story.append(PageBreak())
                    story.append(Paragraph("Model Performance Diagnostics", styles['h2']))
                    story.append(Paragraph(
                        "This section provides diagnostic plots for the machine learning models used to predict player points. "
                        "These plots help assess model performance, stability, and reliability. Understanding these diagnostics is crucial "
                        "for evaluating whether the predictions can be trusted for fantasy football decisions.",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 12))
                    
                    model_diagnostics = self._generate_model_diagnostics_report(models_data, styles)
                    if model_diagnostics:
                        story.extend(model_diagnostics)
                    else:
                        story.append(Paragraph("No model diagnostics available.", styles['Normal']))
                    story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating model diagnostics: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating model diagnostics.", styles['Normal']))
                story.append(PageBreak())

            # 6. Optimization Results and Recommendations
            try:
                if recommendations:
                    story.append(Paragraph("Optimization Results", styles['h2']))
                    story.append(Paragraph(
                        "This section presents the optimized team selection based on the model predictions. "
                        "The optimization considers player form, fixtures, and other key factors to maximize expected points.",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 12))
                    
                    optimization_report = self._generate_optimization_report(recommendations, models_data, X, styles)
                    if optimization_report:
                        story.extend(optimization_report)
                    else:
                        story.append(Paragraph("No optimization results available.", styles['Normal']))
                    story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating optimization report: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating optimization report.", styles['Normal']))
                story.append(PageBreak())

            # 7. Player News Section
            try:
                story.append(Paragraph("Latest Player News", styles['h2']))
                story.append(Paragraph(
                    "This section provides the latest news and updates about players in your squad and recommended players. "
                    "The news is sourced from various football news outlets and is updated daily.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 12))
                
                # Get player news from Perplexity API cache
                player_news = self._get_player_news(recommendations)
                if player_news:
                    story.extend(player_news)
                else:
                    story.append(Paragraph("No recent player news available.", styles['Normal']))
                story.append(PageBreak())
            except Exception as e:
                self.logger.error(f"Error generating player news section: {str(e)}", exc_info=True)
                story.append(Paragraph("Error generating player news section.", styles['Normal']))
                story.append(PageBreak())

            # Build the PDF document
            try:
                doc.build(story)
                self.logger.info(f"Successfully generated report at: {pdf_path}")
                return {
                    'status': 'success',
                    'report_path': pdf_path,
                    'message': 'Report generated successfully.'
                }
            except Exception as e:
                self.logger.error(f"Error building PDF document: {str(e)}", exc_info=True)
                return {
                    'status': 'error',
                    'report_path': None,
                    'message': f'Error building PDF document: {str(e)}'
                }

        except Exception as e:
            self.logger.error(f"Error in report generation: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'report_path': None,
                'message': f'Error in report generation: {str(e)}'
            }
            story.append(Paragraph(
                "<b>Convergence Curves:</b> Show training and validation error over boosting rounds. "
                "<b>Good:</b> Both curves decrease and stabilize (RMSE < 1.5 for FPL points). "
                "<b>Warning:</b> Large gap between train/validation suggests overfitting. "
                "<b>Bad:</b> Validation error increases after initial decrease, or RMSE > 2.5.",
                styles['Normal']
            ))
            story.append(Spacer(1, 6))
            
            story.append(Paragraph(
                "<b>Residual Plots:</b> Show prediction errors vs predicted values. "
                "<b>Good:</b> Random scatter around zero with no clear patterns, residuals within ±3 points. "
                "<b>Warning:</b> Slight curvature or heteroscedasticity (changing variance). "
                "<b>Bad:</b> Clear patterns, systematic bias, or residuals exceeding ±5 points.",
                styles['Normal']
            ))
            story.append(Spacer(1, 6))
            
            story.append(Paragraph(
                "<b>Residual Distribution:</b> Should be approximately normal (bell-shaped) centered at zero. "
                "<b>Good:</b> Symmetric distribution with most residuals within ±2 points. "
                "<b>Warning:</b> Slight skewness or heavy tails. "
                "<b>Bad:</b> Highly skewed, multi-modal, or very heavy tails.",
                styles['Normal']
            ))
            story.append(Spacer(1, 6))
            
            story.append(Paragraph(
                "<b>Q-Q Plots:</b> Compare residual distribution to normal distribution. "
                "<b>Good:</b> Points closely follow the diagonal line. "
                "<b>Warning:</b> Slight deviations at the tails. "
                "<b>Bad:</b> S-shaped curves or major deviations from the line.",
                styles['Normal']
            ))
            story.append(Spacer(1, 12))
            
            story.extend(self._generate_model_diagnostics_report(models_data, styles))
            story.append(PageBreak())

        # 4. Optimization Results
        if recommendations:
            story.append(Paragraph("Optimization Results", styles['h2']))
            story.append(Paragraph(
                "This section presents the optimal Fantasy Premier League squad as determined by a Mixed-Integer Linear Programming (MILP) optimizer. "
                "The squad is selected to maximize total expected points (xP) while adhering to the game's constraints (e.g., budget, team size, and formation). "
                "The 'Key Drivers' column highlights the top three features influencing each player's point prediction, based on SHAP analysis.",
                styles['Normal']
            ))
            story.append(Spacer(1, 12))
            story.extend(self._generate_optimization_report(recommendations, models_data, X, styles))

        doc.build(story)
        self.logger.info(f"Successfully generated report: {pdf_path}")

    def _generate_feature_analysis(self, df: pd.DataFrame, feature_lineage: list, styles) -> list:
        """Analyzes and visualizes feature distributions and lineage."""
        self.logger.debug("Generating feature analysis...")
        flowables = [Paragraph("Input Feature Analysis", styles['h2'])]

        # Feature Statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = df[numeric_cols].describe().transpose().round(2)
        stats.reset_index(inplace=True)
        stats.rename(columns={'index': 'Feature'}, inplace=True)
        stats_data = [stats.columns.tolist()] + stats.values.tolist()
        table = Table(stats_data, hAlign='LEFT', colWidths=[1.2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 0.2*inch))

        # Feature Plots
        for col in numeric_cols:
            if df[col].nunique() > 1:
                buffer = io.BytesIO()
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col], kde=True, bins=30)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                flowables.append(Image(buffer, width=5*inch, height=3*inch))
                flowables.append(Spacer(1, 0.1*inch))

        # Feature Lineage
        flowables.append(PageBreak())
        flowables.append(Paragraph("Feature Lineage", styles['h2']))
        lineage_data = [['Feature', 'Source Fields', 'Transformation']] + [[l['feature'], ', '.join(l['source']) if isinstance(l['source'], list) else l['source'], l['transformation']] for l in feature_lineage]
        lineage_table = Table(lineage_data, colWidths=[1.5*inch, 2*inch, 3.5*inch], hAlign='LEFT')
        lineage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        flowables.append(lineage_table)

        return flowables

    def _generate_boruta_report(self, X: pd.DataFrame, y: pd.Series, styles) -> list:
        """Performs Boruta feature selection and returns results."""
        self.logger.info("Running Boruta feature importance analysis...")
        flowables = [Paragraph("Boruta Feature Importance", styles['h2'])]
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
        boruta_selector.fit(X.values, y.values)

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': boruta_selector.ranking_,
            'Is Important': boruta_selector.support_
        }).sort_values('Ranking')
        
        boruta_data = [importance_df.columns.tolist()] + importance_df.values.tolist()
        table = Table(boruta_data, hAlign='LEFT', colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        flowables.append(table)
        return flowables

    def _generate_model_diagnostics_report(self, models_data: dict, styles) -> list:
        """Generates diagnostics for each model."""
        self.logger.debug("Generating model diagnostics...")
        flowables = [Paragraph("Model Diagnostics", styles['h2'])]

        for model_name, data in models_data.items():
            # Skip non-model entries (e.g., 'diagnostic_comparison') or malformed items
            if not isinstance(data, dict) or not all(k in data for k in ('model', 'X_test', 'y_test', 'history')):
                self.logger.debug(f"Skipping non-model diagnostics entry: {model_name}")
                continue

            flowables.append(Paragraph(f"Diagnostics for: {model_name}", styles['h3']))
            model, X_test, y_test, history = data['model'], data['X_test'], data['y_test'], data['history']

            # Ensure test features align with the model's training features to avoid mismatch
            def _align_features(X: pd.DataFrame, model_obj) -> pd.DataFrame:
                feature_names = getattr(model_obj, 'feature_names_in_', None)
                if feature_names is None:
                    feature_names = getattr(model_obj, 'feature_name_', None)
                if feature_names is None:
                    return X  # best effort
                
                # Convert to list if it's not already
                if not isinstance(feature_names, list):
                    self.logger.warning("Feature names are not a list, attempting conversion")
                    if isinstance(feature_names, np.ndarray):
                        feature_names = feature_names.tolist()
                    
                feature_list = list(feature_names)
                # Use dict comprehension to avoid DataFrame fragmentation
                feature_data = {}
                for f in feature_list:
                    if f in X.columns:
                        feature_data[f] = X[f]
                    else:
                        feature_data[f] = 0.0
                return pd.DataFrame(feature_data, index=X.index)[feature_list]

            X_test_aligned = _align_features(X_test, model)

            # Performance Metrics
            preds = model.predict(X_test_aligned)
            rmse = ((preds - y_test) ** 2).mean() ** 0.5
            mae = (abs(preds - y_test)).mean()
            r2 = model.score(X_test_aligned, y_test)
            metrics_data = [['Metric', 'Value'], ['RMSE', f'{rmse:.3f}'], ['MAE', f'{mae:.3f}'], ['R2', f'{r2:.3f}']]
            metrics_table = Table(metrics_data, hAlign='LEFT')
            flowables.append(metrics_table)
            flowables.append(Spacer(1, 0.2*inch))

            # Convergence Curve
            if history and isinstance(history, dict) and any(history.get(k) for k in ['validation_0', 'validation_1', 'testing']):
                conv_img = self._plot_convergence(history, model_name)
                flowables.append(Image(conv_img, width=5*inch, height=3*inch))
                flowables.append(Spacer(1, 0.1*inch))
            else:
                flowables.append(Paragraph("No convergence history available for this model.", styles['Normal']))
                flowables.append(Spacer(1, 0.1*inch))

            # Residuals Plot (with binned mean overlay)
            res_img = self._plot_residuals(preds, y_test, model_name)
            flowables.append(Image(res_img, width=5*inch, height=3*inch))
            flowables.append(Spacer(1, 0.2*inch))

            # Residuals Histogram
            try:
                res_hist_img = self._plot_residuals_hist(y_test - preds, model_name)
                flowables.append(Image(res_hist_img, width=5*inch, height=3*inch))
                flowables.append(Spacer(1, 0.2*inch))
            except Exception as e:
                self.logger.info(f"Skipping residuals histogram for {model_name}: {e}")

            # Residuals Q-Q Plot (optional if SciPy available)
            try:
                if _scipy_stats is not None:
                    qq_img = self._plot_residuals_qq(y_test - preds, model_name)
                    flowables.append(Image(qq_img, width=5*inch, height=3*inch))
                    flowables.append(Spacer(1, 0.2*inch))
            except Exception as e:
                self.logger.info(f"Skipping residuals Q-Q plot for {model_name}: {e}")

            # SHAP Summary Plot (inline if available)
            try:
                if shap is not None:
                    explainer = data.get('explainer')
                    shap_values = data.get('shap_values')
                    if explainer is None or shap_values is None:
                        # best-effort computation if missing
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test_aligned)
                        except Exception:
                            explainer = None
                            shap_values = None
                    if shap_values is not None:
                        plt.figure(figsize=(8, 5))
                        shap.summary_plot(shap_values, X_test_aligned, show=False)
                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        plt.close()
                        buf.seek(0)
                        flowables.append(Paragraph(f"SHAP Summary for {model_name}", styles['h4']))
                        flowables.append(Image(buf, width=6*inch, height=4*inch))
                        flowables.append(Spacer(1, 0.2*inch))
                elif data.get('shap_plot_path') and os.path.exists(data['shap_plot_path']):
                    # fallback to pre-generated image if SHAP not imported here
                    flowables.append(Paragraph(f"SHAP Summary for {model_name}", styles['h4']))
                    flowables.append(Image(data['shap_plot_path'], width=6*inch, height=4*inch))
            except Exception as shap_e:
                self.logger.info(f"Skipping SHAP summary for {model_name}: {shap_e}")

            flowables.append(PageBreak())

        return flowables

    def _plot_convergence(self, history: dict, model_name: str) -> io.BytesIO:
        """Plots convergence curve and returns an image buffer."""
        buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        plotted_any = False
        if 'validation_0' in history and 'rmse' in history['validation_0']:
            plt.plot(history['validation_0']['rmse'], label='Train RMSE')
            plotted_any = True
        if 'validation_1' in history and 'rmse' in history['validation_1']:
            plt.plot(history['validation_1']['rmse'], label='Test RMSE')
            plotted_any = True
        elif 'testing' in history and 'rmse' in history['testing']:
            plt.plot(history['testing']['rmse'], label='Test RMSE')
            plotted_any = True

        plt.title(f'Convergence for {model_name}')
        plt.xlabel('Boosting Round'); plt.ylabel('RMSE')
        if plotted_any:
            plt.legend()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    def _plot_residuals(self, preds: pd.Series, y_test: pd.Series, model_name: str) -> io.BytesIO:
        """Plots residuals with binned mean overlay and returns an image buffer."""
        buffer = io.BytesIO()
        residuals = y_test - preds
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=preds, y=residuals, alpha=0.5)
        # Binned mean residual overlay
        try:
            bins = np.linspace(np.nanmin(preds), np.nanmax(preds), 15)
            inds = np.digitize(preds, bins)
            bin_centers = []
            bin_means = []
            for b in range(1, len(bins)):
                mask = inds == b
                if np.any(mask):
                    bin_centers.append(np.nanmean(preds[mask]))
                    bin_means.append(np.nanmean(residuals[mask]))
            if bin_centers:
                plt.plot(bin_centers, bin_means, color='orange', linewidth=2, label='Binned mean residual')
                plt.legend()
        except Exception:
            pass
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals vs. Predicted for {model_name}')
        plt.xlabel('Predicted Values'); plt.ylabel('Residuals')
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    def _plot_residuals_qq(self, residuals: pd.Series, model_name: str) -> io.BytesIO:
        """Plots a Q-Q plot of residuals if SciPy is available."""
        if _scipy_stats is None:
            raise RuntimeError("SciPy not available for Q-Q plot")
        buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        _scipy_stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'Residuals Q-Q Plot for {model_name}')
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    def _plot_residuals_hist(self, residuals: pd.Series, model_name: str) -> io.BytesIO:
        """Plots residuals histogram with KDE and returns an image buffer."""
        buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f'Residuals Distribution for {model_name}')
        plt.xlabel('Residual'); plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    def _generate_html(self, data: dict) -> str:
        """Render the report data into an HTML file."""
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template('report_template.html')
        
        html_content = template.render(
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            features=data.get('features', []),
            boruta_plot_path=data.get('boruta_plot_path', ''),
            models=data.get('models', [])
        )
        
        html_path = os.path.join(self.report_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return html_path

    def _convert_html_to_pdf(self, html_path: str, pdf_path: str):
        """Convert an HTML file to a PDF using WeasyPrint."""
        HTML(html_path).write_pdf(pdf_path)

    def _generate_optimization_report(self, recommendations: dict, models_data: dict, X: pd.DataFrame, styles) -> list:
        """Generates a report section for the final optimization results."""
        self.logger.debug("Generating optimization results report...")
        flowables = [Paragraph("Optimization Results", styles['h2'])]

        if not recommendations or 'optimal_team' not in recommendations:
            flowables.append(Paragraph("No optimization data available.", styles['Normal']))
            return flowables

        team = recommendations['optimal_team']
        captaincy = recommendations['captaincy']

        # Handle the case where captaincy might be None
        if captaincy is None:
            captaincy = {}

        # Summary Table
        summary_data = [
            ['Metric', 'Value'],
            ['Formation', team.get('formation', 'N/A')],
            ['Expected Points', f"{team.get('expected_points', 0):.2f}"],
            ['Total Cost', f"£{team.get('total_cost', 0) / 10:.1f}m"],
            ['Captain', captaincy.get('captain', {}).get('web_name', 'N/A') if captaincy.get('captain') else 'N/A'],
            ['Vice-Captain', captaincy.get('vice_captain', {}).get('web_name', 'N/A') if captaincy.get('vice_captain') else 'N/A']
        ]
        summary_table = Table(summary_data, hAlign='LEFT', colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        flowables.append(summary_table)
        flowables.append(Spacer(1, 0.2*inch))

        # Starting XI Table
        flowables.append(Paragraph("Starting XI", styles['h3']))
        starting_xi = team.get('starting_xi', [])
        xi_data = [['Player', 'Position', 'Team', 'Cost (£m)', 'xP', 'Key Drivers']]
        for p in starting_xi:
            drivers = self._get_player_shap_drivers(p, models_data, X)
            xi_data.append([p['web_name'], p['position'], p.get('team_name', 'N/A'), f"{p['now_cost'] / 10:.1f}", f"{p.get('predicted_points', 0):.1f}", drivers])
        
        xi_table = Table(xi_data, hAlign='LEFT', colWidths=[1.2*inch, 0.7*inch, 1.0*inch, 0.7*inch, 0.5*inch, 2.9*inch])
        xi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        flowables.append(xi_table)
        flowables.append(Spacer(1, 0.2*inch))

        # Bench Players Table
        flowables.append(Paragraph("Bench", styles['h3']))
        bench_players = team.get('bench', [])
        if bench_players:
            bench_data = [['Player', 'Position', 'Team', 'Cost (£m)', 'xP', 'Key Drivers']]
            for p in bench_players:
                drivers = self._get_player_shap_drivers(p, models_data, X)
                bench_data.append([p['web_name'], p['position'], p.get('team_name', 'N/A'), f"{p['now_cost'] / 10:.1f}", f"{p.get('predicted_points', 0):.1f}", drivers])
            
            bench_table = Table(bench_data, hAlign='LEFT', colWidths=[1.2*inch, 0.7*inch, 1.0*inch, 0.7*inch, 0.5*inch, 2.9*inch])
            bench_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 8)
            ]))
            flowables.append(bench_table)
            flowables.append(Spacer(1, 0.2*inch))

        return flowables

    def _get_player_shap_drivers(self, player: dict, models_data: dict, X: pd.DataFrame) -> str:
        """Calculate and format top 3 SHAP drivers for a single player."""
        try:
            # Check if SHAP is available
            if shap is None:
                return "SHAP not available"
            
            # Map position from element_type or position string
            if 'element_type' in player:
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                position = position_map.get(player['element_type'])
            else:
                position = player.get('position')
            
            if not position:
                return "Position unknown"

            # Try both model types, prefer the one with SHAP data
            for model_suffix in ['_4_lgbm', '_4_xgb']:
                model_key = f'{position}{model_suffix}'
                if model_key in models_data:
                    model_info = models_data[model_key]
                    model = model_info['model']
                    
                    # Try to use existing SHAP values first
                    if model_info.get('shap_values') is not None and model_info.get('X_test') is not None:
                        X_test = model_info['X_test']
                        shap_values = model_info['shap_values']
                        
                        # Use mean absolute SHAP values across test set as proxy
                        if len(shap_values.shape) > 1:
                            mean_shap = np.abs(shap_values).mean(axis=0)
                        else:
                            mean_shap = np.abs(shap_values)
                        
                        feature_names = X_test.columns
                        top_indices = np.argsort(mean_shap)[-3:][::-1]
                        
                        drivers_str = []
                        for idx in top_indices:
                            feature = feature_names[idx]
                            value = mean_shap[idx]
                            drivers_str.append(f"{feature} ({value:.2f})")
                        
                        return ", ".join(drivers_str)
                    
                    # Fallback: try to compute SHAP for this specific player
                    try:
                        player_id = player.get('id')
                        if player_id is not None and player_id in X.index:
                            player_features = X.loc[player_id:player_id]  # Keep as DataFrame
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(player_features)
                            
                            if len(shap_values.shape) > 1:
                                shap_values = shap_values[0]
                            
                            feature_names = X.columns
                            top_indices = np.argsort(np.abs(shap_values))[-3:][::-1]
                            
                            drivers_str = []
                            for idx in top_indices:
                                feature = feature_names[idx]
                                value = shap_values[idx]
                                direction = "+" if value > 0 else "-"
                                drivers_str.append(f"{feature} ({direction}{abs(value):.2f})")
                            
                            return ", ".join(drivers_str)
                    except Exception:
                        continue
            
            return "SHAP unavailable"

        except Exception as e:
            self.logger.debug(f"Could not generate SHAP drivers for player {player.get('web_name', 'N/A')}: {e}")
            return "Feature analysis unavailable"

    def _generate_cleaning_report(self, cleaned_stats: dict, styles) -> list:
        """Generates a report section for data cleaning statistics."""
        self.logger.debug("Generating data cleaning report...")
        flowables = [Paragraph("Data Cleaning and Clipping Report", styles['h2'])]

        if not cleaned_stats:
            flowables.append(Paragraph("No cleaning statistics available.", styles['Normal']))
            return flowables

        stats_data = [['Feature', 'Clipped Values', 'Total Values', '% Clipped']]
        for feature, stats in cleaned_stats.items():
            clipped_count = stats.get('clipped_count', 0)
            total_count = stats.get('total_count', 1)
            percentage = (clipped_count / total_count) * 100 if total_count > 0 else 0
            stats_data.append([feature, clipped_count, total_count, f"{percentage:.2f}%"])

        table = Table(stats_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        flowables.append(table)
        return flowables
    
    def _get_player_news(self, recommendations: dict) -> list:
        """
        Fetches and formats player news from the Perplexity API cache.
        
        Args:
            recommendations: Dictionary containing optimization results and recommendations
            
        Returns:
            list: List of ReportLab flowables for the player news section
        """
        try:
            self.logger.info("Fetching player news from cache...")
            flowables = []
            
            # Check if we have a cache directory for Perplexity API responses
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  '.perplexity_cache')
            
            if not os.path.exists(cache_dir):
                self.logger.warning(f"Perplexity cache directory not found: {cache_dir}")
                return []
                
            # Get the most recent cache file (sorted by modification time)
            cache_files = []
            for root, _, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.json'):
                        cache_files.append(os.path.join(root, file))
            
            if not cache_files:
                self.logger.warning("No cache files found in the Perplexity cache directory.")
                return []
                
            # Sort by modification time (newest first)
            cache_files.sort(key=os.path.getmtime, reverse=True)
            latest_cache = cache_files[0]
            
            self.logger.info(f"Loading player news from cache: {latest_cache}")
            
            # Load the cached news data
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache file {latest_cache}: {str(e)}")
                return []
            
            # Extract player IDs from recommendations
            player_ids = set()
            if recommendations and 'optimal_team' in recommendations:
                team = recommendations['optimal_team']
                if 'squad' in team:
                    for player in team['squad']:
                        if 'id' in player:
                            player_ids.add(str(player['id']))
            
            if not player_ids:
                self.logger.warning("No player IDs found in recommendations to fetch news for.")
                return []
            
            # Filter news for players in our team
            player_news = {}
            for player_id, news_items in news_data.items():
                if player_id in player_ids and news_items:
                    player_news[player_id] = news_items
            
            if not player_news:
                self.logger.info("No news found for players in the current team.")
                return []
            
            # Format the news for the report
            styles = getSampleStyleSheet()
            flowables = []
            
            for player_id, news_items in player_news.items():
                # Get player name from recommendations
                player_name = f"Player {player_id}"
                if recommendations and 'optimal_team' in recommendations:
                    for player in recommendations['optimal_team'].get('squad', []):
                        if str(player.get('id')) == player_id:
                            player_name = player.get('web_name', player_name)
                            break
                
                # Add player header
                flowables.append(Paragraph(f"<b>{player_name}</b>", styles['h3']))
                
                # Add each news item
                for item in news_items[:5]:  # Limit to 5 most recent news items per player
                    # Format the date if available
                    date_str = ""
                    if 'date' in item:
                        try:
                            date_obj = datetime.strptime(item['date'], '%Y-%m-%dT%H:%M:%S')
                            date_str = date_obj.strftime('%b %d, %Y')
                        except (ValueError, TypeError):
                            date_str = item.get('date', '')
                    
                    # Add the news item with formatting
                    news_text = f"<b>{date_str}:</b> {item.get('title', 'No title')}"
                    if 'summary' in item and item['summary']:
                        news_text += f"<br/>&nbsp;&nbsp;&nbsp;&nbsp;{item['summary']}"
                    
                    flowables.append(Paragraph(news_text, styles['Normal']))
                    flowables.append(Spacer(1, 6))
                
                flowables.append(Spacer(1, 12))
            
            return flowables
            
        except Exception as e:
            self.logger.error(f"Error in _get_player_news: {str(e)}", exc_info=True)
            return []

    def _generate_diagnostic_comparison_report(self, diagnostic_comparison: dict, styles) -> list:
        """Generates a report section for XGBoost vs Decision Tree model comparison."""
        self.logger.debug("Generating diagnostic comparison report...")
        flowables = [Paragraph("XGBoost vs Decision Tree Diagnostic Comparison", styles['h3'])]
        
        # Add explanatory text
        flowables.append(Paragraph(
            "The Decision Tree model serves as a diagnostic baseline to assess XGBoost performance. "
            "The tree is limited to depth 2 for interpretability. Lower RMSE indicates better accuracy.",
            styles['Normal']
        ))
        flowables.append(Spacer(1, 12))
        
        # Create comparison table for each position
        for position, comparison_data in diagnostic_comparison.items():
            flowables.append(Paragraph(f"{position} Model Comparison", styles['h4']))
            
            # Build table data
            table_data = [['Model', 'RMSE', 'MAE', 'R²']]
            for i, model_name in enumerate(comparison_data['Model']):
                rmse = f"{comparison_data['RMSE'][i]:.3f}" if comparison_data['RMSE'][i] is not None else "N/A"
                mae = f"{comparison_data['MAE'][i]:.3f}" if comparison_data['MAE'][i] is not None else "N/A"
                r2 = f"{comparison_data['R²'][i]:.3f}" if comparison_data['R²'][i] is not None else "N/A"
                table_data.append([model_name, rmse, mae, r2])
            
            comparison_table = Table(table_data, hAlign='LEFT', colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
            comparison_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            flowables.append(comparison_table)
            flowables.append(Spacer(1, 12))
            
            # Add performance interpretation
            if comparison_data['RMSE'][0] is not None and comparison_data['RMSE'][1] is not None:
                xgb_rmse = comparison_data['RMSE'][0]
                dt_rmse = comparison_data['RMSE'][1]
                improvement = ((dt_rmse - xgb_rmse) / dt_rmse) * 100
                
                interpretation = f"XGBoost shows {improvement:+.1f}% performance vs Decision Tree baseline"
                if improvement > 20:
                    interpretation += " (Strong improvement - model is learning complex patterns)"
                elif improvement > 10:
                    interpretation += " (Good improvement - model is effective)"
                elif improvement > 0:
                    interpretation += " (Modest improvement - consider feature engineering)"
                else:
                    interpretation += " (Warning - may indicate overfitting or data issues)"
                    
                flowables.append(Paragraph(interpretation, styles['Normal']))
                flowables.append(Spacer(1, 12))
        
        return flowables

    def generate_report(self, df: pd.DataFrame, feature_lineage: list, models_data: dict, recommendations: dict, X: pd.DataFrame) -> dict:
        """Generate a comprehensive PDF report with all analyses and return its path.
        
        Returns a dictionary with key 'report_path' pointing to the PDF generated by
        generate_full_report(). Falls back to the most recent matching PDF in the
        reports directory if the direct result does not contain a path.
        """
        # Extract the target variable y from the data (assume 'total_points' when available)
        if 'total_points' in df.columns:
            y = df['total_points']
        else:
            # Create an empty Series aligned with df index as a safe fallback
            y = pd.Series(index=df.index)
        
        # Extract cleaned_stats from the data if available (placeholder for now)
        cleaned_stats = {}
        
        # Generate the full report and use its returned path
        result = self.generate_full_report(df, feature_lineage, models_data, X, y, cleaned_stats, recommendations)
        if isinstance(result, dict) and result.get('report_path'):
            return {'report_path': result['report_path']}
        
        # Fallback: find the most recent matching report in the report directory
        try:
            candidates = [
                f for f in os.listdir(self.report_dir)
                if f.startswith('fantasy_football_report_') and f.endswith('.pdf')
            ]
            if candidates:
                latest = max(candidates, key=lambda f: os.path.getmtime(os.path.join(self.report_dir, f)))
                return {'report_path': os.path.join(self.report_dir, latest)}
        except Exception:
            pass
        
        # If all else fails, return a None path to indicate failure to resolve
        return {'report_path': None}
