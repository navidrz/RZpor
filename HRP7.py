# hrp_streamlit_app_updated_with_SpearmanDistance_and_DistanceCorrelation.py

# %% (No changes to initial imports)
# =================================
# Hierarchical Risk Parity, HERC & NCO with Spearman & Distance Correlation - Streamlit App (Enhanced Version)
# =================================

"""
این برنامه استریم‌لیت، بهینه‌سازی Hierarchical Risk Parity (HRP)، Hierarchical Equal Risk Contribution (HERC) و Nested Clusters Optimization (NCO) را با استفاده از متغیرهای وابسته (y) و فاصله‌سنجی Spearman و Distance Correlation بهبود می‌بخشد. این برنامه به شما امکان می‌دهد داده‌های مورد نیاز را بارگذاری کرده، پارامترهای مدل را تنظیم کنید و پورتفوی بهینه را به صورت پویا مشاهده و تحلیل کنید.
"""

# %% (No changes to imports, except ensure httpx is imported)
import asyncio
import logging
from datetime import datetime
import sys
import re
import pandas as pd
import numpy as np
import streamlit as st
from fuzzywuzzy import fuzz, process
from plotly.io import to_html
from sklearn.model_selection import train_test_split
from tsetmc_api.symbol import Symbol
import httpx  # Added to customize requests

from skfolio import Population, RiskMeasure, RatioMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import KendallDistance, SpearmanDistance, DistanceCorrelation
from skfolio.optimization import EqualWeighted, HierarchicalRiskParity, DistributionallyRobustCVaR, HierarchicalEqualRiskContribution, NestedClustersOptimization
from skfolio.prior import FactorModel

import jdatetime
import base64
from io import BytesIO

# %%
# Configure Logging
# =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Utility Functions
# =================
def validate_symbol_ids(symbol_ids: list) -> bool:
    """Validates that all symbol IDs are numeric."""
    pattern = re.compile(r'^\d+$')
    return all(pattern.match(id_) for id_ in symbol_ids)

def get_table_download_link(df, filename, link_text):
    """
    Generates a download link for a pandas DataFrame.
    """
    try:
        if filename.endswith('.csv'):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        elif filename.endswith('.xlsx'):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            xlsx_data = output.getvalue()
            b64 = base64.b64encode(xlsx_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
        else:
            href = "#"
        return href
    except Exception as e:
        logger.error(f"Error generating download link: {e}")
        return "#"

# %% 
# DataFetcher Class
# =================
class DataFetcher:
    """Class responsible for fetching and processing data from the API."""

    @staticmethod
    async def fetch_daily_history(symbol_id: str) -> pd.DataFrame:
        """
        Fetches the daily history of a symbol asynchronously with a custom User-Agent header 
        to reduce the chance of IP-based or headless request restrictions.

        Parameters:
        -----------
        symbol_id : str
            The unique identifier for the symbol.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing 'date', 'close', 'volume', 'value' columns.
        """
        try:
            logger.info(f"Start fetching daily history for symbol_id: {symbol_id}")

            # Custom headers to mimic a real browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            # Create a custom httpx.AsyncClient with the specified headers
            async with httpx.AsyncClient(headers=headers, timeout=30) as client:
                symbol = Symbol(symbol_id=symbol_id)
                # If Symbol or Symbol class has a _http_client attribute you can overwrite:
                if hasattr(symbol, '_http_client'):
                    symbol._http_client = client
                # Now call the async method
                daily_history = await symbol.get_daily_history_async()


            if not daily_history:
                logger.warning(f"No daily history data retrieved for symbol_id: {symbol_id}")
                return pd.DataFrame()

            stock_data = []
            for row in daily_history:
                try:
                    # Convert Jalali dates to Gregorian if necessary
                    if isinstance(row.date, jdatetime.date):
                        gregorian_date = pd.to_datetime(row.date.togregorian()).normalize()
                    elif isinstance(row.date, datetime):
                        gregorian_date = pd.to_datetime(row.date).normalize()
                    else:
                        gregorian_date = pd.NaT
                        logger.warning(f"Unknown date format for row: {row}")

                    stock_data.append({
                        'date': gregorian_date,
                        'close': row.close,
                        'volume': row.volume,
                        'value': row.value
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid date value: {row.date} - {e}")

            stock_df = pd.DataFrame(stock_data).dropna(subset=['date'])
            stock_df = stock_df.sort_values('date').drop_duplicates(subset='date')  # Remove duplicate dates
            stock_df['close'] = pd.to_numeric(stock_df['close'], errors='coerce')
            stock_df = stock_df.dropna(subset=['close'])
            logger.info(f"Successfully fetched and converted daily history for symbol_id: {symbol_id}")
            if not stock_df.empty:
                logger.info(f"Date Range for symbol_id {symbol_id}: {stock_df['date'].min()} to {stock_df['date'].max()}")
            return stock_df
        except Exception as e:
            logger.error(f"Error fetching daily history for symbol_id: {symbol_id} - {e}")
            return pd.DataFrame()

    @staticmethod
    async def fetch_all_symbols(symbol_ids: list) -> dict:
        """
        Fetches daily history data for all symbols asynchronously using the custom headers.

        Parameters:
        -----------
        symbol_ids : list
            List of symbol IDs to fetch data for.

        Returns:
        --------
        dict
            Dictionary mapping symbol IDs to their fetched DataFrames.
        """
        tasks = [DataFetcher.fetch_daily_history(symbol_id) for symbol_id in symbol_ids]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbol_ids, results))

# %%
# DataLoader Class
# ================
class DataLoader:
    """Class responsible for loading and processing external data files."""

    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list, file_type: str) -> bool:
        """
        Validates that the DataFrame contains all required columns.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate.
        required_columns : list
            List of required column names.
        file_type : str
            Type of data file.

        Returns:
        --------
        bool
            True if all required columns are present, False otherwise.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns {missing} in {file_type} data.")
            return False
        return True

    @staticmethod
    def load_data(file_type: str, file_content) -> pd.DataFrame:
        """
        Loads data from uploaded file content.

        Parameters:
        -----------
        file_type : str
            Type of data to load ('market', 'risk_free_rate', 'market_cap', 'usd_to_rial').
        file_content : BytesIO
            Uploaded file content.

        Returns:
        --------
        pd.DataFrame
            Loaded and processed DataFrame with standardized column names.
        """
        try:
            # Determine file extension
            file_extension = file_content.name.split('.')[-1].lower()

            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_content, index_col=None)
            elif file_extension == 'csv':
                df = pd.read_csv(file_content, index_col=None)
            else:
                logger.error("Unsupported file format uploaded.")
                return pd.DataFrame()

            logger.info(f"Loaded {file_type} data from uploaded file.")

            # Define required fields based on file type
            required_fields = {}
            if file_type == 'market':
                required_fields = {
                    'date': ['date', 'Date', 'Gregorian Date', 'تاریخ میلادی'],
                    'daily_return': ['return', 'daily return', 'بازده', 'بازده روزانه'],
                }
            elif file_type == 'risk_free_rate':
                required_fields = {
                    'date': ['date', 'Date', 'Gregorian Date', 'تاریخ میلادی'],
                    'risk_free_rate': ['ytm', 'YTM', 'Yield to Maturity', 'Interest Rate', 'Risk-Free Rate'],
                }
            elif file_type == 'market_cap':
                required_fields = {
                    'date': ['date', 'Date', 'Gregorian Date', 'تاریخ میلادی'],
                    'market_cap': ['market_cap', 'Market Cap', 'Market Capitalization', 'بازار سرمایه', 'price'],
                }
            elif file_type == 'usd_to_rial':
                required_fields = {
                    'date': ['date', 'Date', 'Gregorian Date', 'تاریخ میلادی'],
                    'usd_to_rial': ['usd_to_rial', 'USD to Rial', 'Exchange Rate', 'نرخ تبدیل دلار به ریال', 'price'],
                }

            # Normalize column names
            normalized_columns = {col.strip().lower(): col for col in df.columns}
            mapped_columns = {}

            for field, possible_names in required_fields.items():
                # Attempt to find the best match for the field
                best_match, score = process.extractOne(field, list(normalized_columns.keys()), scorer=fuzz.token_sort_ratio)
                if score >= 90:
                    original_col = normalized_columns[best_match]
                    mapped_columns[field] = original_col
                    logger.info(f"Mapped '{field}' to column '{original_col}' with score {score}")
                else:
                    # Try alternative names
                    best_alt_match = None
                    best_alt_score = 0
                    for alt_name in possible_names:
                        alt_name_normalized = alt_name.strip().lower()
                        match, alt_score = process.extractOne(
                            query=alt_name_normalized,
                            choices=list(normalized_columns.keys()),
                            scorer=fuzz.token_sort_ratio
                        )
                        if alt_score > best_alt_score and alt_score >= 90:
                            best_alt_match = normalized_columns.get(match)
                            best_alt_score = alt_score
                    if best_alt_match:
                        mapped_columns[field] = best_alt_match
                        logger.info(f"Mapped '{field}' to column '{best_alt_match}' with score {best_alt_score}")
                    else:
                        logger.error(f"Required column for '{field}' not found in {file_type} data.")
                        return pd.DataFrame()

            # Rename columns to standardized names
            rename_mapping = {mapped_columns[field]: field for field in required_fields}
            df = df.rename(columns=rename_mapping)

            # Validate required columns
            if not DataLoader.validate_data(df, required_fields.keys(), file_type):
                return pd.DataFrame()

            # Convert 'date' column to datetime and normalize to remove time components
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').drop_duplicates(subset='date')  # Remove duplicate dates

            # Log the date range
            start_date = df['date'].min()
            end_date = df['date'].max()
            logger.info(f"{file_type.capitalize()} Data Date Range: {start_date} to {end_date}")

            # For numeric columns, ensure correct data types
            numeric_fields = [field for field in required_fields if field != 'date']
            for field in numeric_fields:
                df[field] = pd.to_numeric(df[field], errors='coerce')

            df = df.dropna(subset=numeric_fields)  # Drop rows with NaNs in required numeric fields
            logger.info(f"Successfully loaded and processed {file_type} data.")
            return df
        except Exception as e:
            logger.error(f"Error data loader: {e}")
            return pd.DataFrame()

# %%
# Preprocessor Class
# ==================
class Preprocessor:
    """Handles data preprocessing steps such as calculating returns and aligning datasets."""

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates daily returns from price data.

        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame containing daily closing prices for each stock.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing daily returns.
        """
        try:
            returns = prices.pct_change().dropna()
            logger.info("Calculated daily returns from price data.")
            return returns
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()

    @staticmethod
    def align_datasets(*datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns multiple datasets on their date indices.

        Parameters:
        -----------
        datasets : pd.DataFrame
            Variable number of DataFrames to align.

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with aligned dates.
        """
        try:
            # Log date ranges and sample dates for each dataset
            for i, df in enumerate(datasets):
                if isinstance(df, pd.Series):
                    logger.info(f"Dataset {i}: {df.name} Date Range: {df.index.min()} to {df.index.max()}")
                else:
                    logger.info(f"Dataset {i}: Date Range: {df.index.min()} to {df.index.max()}")
                    # Log sample dates
                    sample_dates = df.index[:5].tolist()
                    logger.info(f"Dataset {i} sample dates: {sample_dates}")

            # Find intersection of all date indices
            common_dates = datasets[0].index
            for df in datasets[1:]:
                common_dates = common_dates.intersection(df.index)

            logger.info(f"Number of common dates after intersection: {len(common_dates)}")

            if len(common_dates) == 0:
                logger.error("No overlapping dates found across datasets after intersection.")
                return pd.DataFrame()

            # Align datasets on the common dates
            combined = pd.concat([df.loc[common_dates] for df in datasets], axis=1)
            combined = combined.dropna()
            logger.info(f"Number of dates after dropna: {len(combined)}")
            logger.info(f"Number of common dates after alignment: {len(combined)}")
            return combined
        except Exception as e:
            logger.error(f"Error aligning datasets: {e}")
            return pd.DataFrame()

    @staticmethod
    def process_data(prices: pd.DataFrame, market_returns: pd.Series, risk_free_rate: pd.Series,
                    market_cap: pd.Series, usd_to_rial: pd.Series) -> tuple:
        """
        Processes and aligns all input data for modeling.

        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame containing daily closing prices.
        market_returns : pd.Series
            Series containing market index returns.
        risk_free_rate : pd.Series
            Series containing risk-free rate data.
        market_cap : pd.Series
            Series containing market capitalization data.
        usd_to_rial : pd.Series
            Series containing USD to Rial exchange rates.

        Returns:
        --------
        tuple
            Tuple containing X_train, X_test, y_train, y_test DataFrames/Series.
        """
        try:
            # Calculate returns
            stock_returns = Preprocessor.calculate_returns(prices)

            # Align all datasets
            combined = Preprocessor.align_datasets(
                stock_returns,
                market_returns,
                risk_free_rate,
                market_cap,
                usd_to_rial
            )

            if combined.empty:
                logger.error("Combined dataset is empty after alignment.")
                return None, None, None, None

            # Define feature set (X) and target variables (y)
            X = stock_returns.loc[combined.index]
            y_excess = market_returns.loc[combined.index] - risk_free_rate.loc[combined.index]
            y_market_cap_change = market_cap.loc[combined.index].pct_change().bfill()
            y_usd_to_rial_change = usd_to_rial.loc[combined.index].pct_change().bfill()

            # Create y DataFrame with multiple factors
            y = pd.DataFrame({
                'excess_return': y_excess,
                'market_cap_change': y_market_cap_change,
                'usd_to_rial_change': y_usd_to_rial_change
            }).dropna()

            # Align X and y
            combined_final = X.loc[y.index]
            if combined_final.empty:
                logger.error("No overlapping data between X and y after processing.")
                return None, None, None, None

            # Split Data into Training and Testing Sets (e.g., 67-33 split)
            X_train, X_test, y_train, y_test = train_test_split(
                combined_final, y, test_size=0.33, shuffle=False
            )
            logger.info(f"Data split into training (size={X_train.shape}) and testing (size={X_test.shape}) sets.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            return None, None, None, None

# %%
# OptimizerModel Class
# ====================
class OptimizerModel:
    """Encapsulates the optimization model creation, fitting, and prediction."""

    def __init__(self, optimizer, name="Optimizer-Model", risk_measure=RiskMeasure.CVAR):
        """
        Initializes the OptimizerModel with the specified optimizer.

        Parameters:
        -----------
        optimizer : skfolio.optimization.BaseRiskParity
            An instance of an optimizer from skfolio.optimization (e.g., HierarchicalRiskParity, DistributionallyRobustCVaR, HierarchicalEqualRiskContribution, NestedClustersOptimization).
        name : str, optional
            Name of the optimizer model (default is "Optimizer-Model").
        risk_measure : RiskMeasure, optional
            The risk measure to use for optimization (default is CVaR).
        """
        self.name = name
        self.optimizer = optimizer
        self.risk_measure = risk_measure

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame = None):
        """
        Fits the optimizer model on the training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training data returns.
        y_train : pd.DataFrame, optional
            Training data factors (default is None).
        """
        try:
            if y_train is not None and hasattr(self.optimizer, 'prior_estimator') and self.optimizer.prior_estimator is not None:
                self.optimizer.fit(X_train, y_train)
            else:
                self.optimizer.fit(X_train)
            logger.info(f"Fitted model '{self.name}'. Weights: {self.optimizer.weights_}")

            # Fit the hierarchical clustering estimator with the correlation matrix
            if hasattr(self.optimizer, 'hierarchical_clustering_estimator') and self.optimizer.hierarchical_clustering_estimator is not None:
                # Compute the correlation matrix
                corr_matrix = X_train.corr()

                # Compute the distance matrix based on the distance estimator
                if isinstance(self.optimizer.distance_estimator, KendallDistance):
                    # Compute Kendall tau correlation
                    kendall_corr = X_train.corr(method='kendall')
                    # Convert correlation to distance
                    distance_matrix = 1 - kendall_corr
                elif isinstance(self.optimizer.distance_estimator, SpearmanDistance):
                    # Compute Spearman correlation
                    spearman_corr = X_train.corr(method='spearman')
                    # Apply power and/or absolute transformations
                    if self.optimizer.distance_estimator.absolute:
                        spearman_corr = spearman_corr.abs()
                    if self.optimizer.distance_estimator.power != 1:
                        spearman_corr = spearman_corr ** self.optimizer.distance_estimator.power
                    # Convert correlation to distance
                    distance_matrix = 1 - spearman_corr
                elif isinstance(self.optimizer.distance_estimator, DistanceCorrelation):
                    # Fit DistanceCorrelation estimator
                    self.optimizer.distance_estimator.fit(X_train)
                    distance_matrix = self.optimizer.distance_estimator.distance_ndarray
                else:
                    # Default to Pearson distance (1 - correlation)
                    distance_matrix = 1 - corr_matrix

                # Fit the hierarchical clustering estimator with the distance matrix
                self.optimizer.hierarchical_clustering_estimator.fit(distance_matrix)
                logger.info(f"Fitted hierarchical clustering estimator for model '{self.name}'.")
        except Exception as e:
            logger.error(f"Error fitting model '{self.name}': {e}")

    def predict(self, X: pd.DataFrame) -> object:
        """
        Predicts the portfolio based on the fitted optimizer.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame of returns to predict on.

        Returns:
        --------
        Portfolio | Population
            Predicted portfolio or population of portfolios.
        """
        try:
            prediction = self.optimizer.predict(X)
            logger.info(f"Predicted portfolio for model '{self.name}'.")
            return prediction
        except Exception as e:
            logger.error(f"Error predicting portfolio for model '{self.name}': {e}")
            return None

    def plot_dendrogram(self, heatmap=False):
        """
        Plots the dendrogram of the hierarchical clustering if supported by the optimizer.

        Parameters:
        -----------
        heatmap : bool, optional
            Whether to display the heatmap of the reordered distance matrix (default is False).
        """
        try:
            if hasattr(self.optimizer, 'hierarchical_clustering_estimator') and \
               hasattr(self.optimizer.hierarchical_clustering_estimator, 'plot_dendrogram'):
                fig = self.optimizer.hierarchical_clustering_estimator.plot_dendrogram(heatmap=heatmap)
                st.plotly_chart(fig, use_container_width=True)
                logger.info(f"Plotted dendrogram for model '{self.name}'.")
            else:
                logger.warning(f"Optimizer '{self.name}' does not support dendrogram plotting.")
        except Exception as e:
            logger.error(f"Error plotting dendrogram for model '{self.name}': {e}")

# %%
# Evaluator Class
# ================
class Evaluator:
    """Handles evaluation of models, including risk contributions, dendrograms, and summary statistics."""

    @staticmethod
    def analyze_risk_contribution(portfolio, measure=RiskMeasure.CVAR):
        """
        Analyzes and plots the risk contribution of the portfolio.

        Parameters:
        -----------
        portfolio : Portfolio | Population
            The portfolio or population of portfolios to analyze.
        measure : RiskMeasure, optional
            The risk measure to use for analysis (default is CVaR).
        """
        try:
            fig = portfolio.plot_contribution(measure=measure)
            st.plotly_chart(fig, use_container_width=True)
            logger.info(f"Plotted risk contribution using {measure}.")
        except Exception as e:
            logger.error(f"Error plotting risk contribution: {e}")

    @staticmethod
    def plot_cumulative_returns(population: Population):
        """
        Plots the cumulative returns of the population of portfolios.

        Parameters:
        -----------
        population : Population
            The population containing multiple portfolios.
        """
        try:
            fig = population.plot_cumulative_returns()
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Plotted cumulative returns for the population.")
        except Exception as e:
            logger.error(f"Error plotting cumulative returns: {e}")

    @staticmethod
    def plot_composition(population: Population):
        """
        Plots the composition of the portfolios in the population.

        Parameters:
        -----------
        population : Population
            The population containing multiple portfolios.
        """
        try:
            fig = population.plot_composition()
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Plotted composition for the population.")
        except Exception as e:
            logger.error(f"Error plotting composition: {e}")

    @staticmethod
    def print_summary(population: Population):
        """
        Displays the summary statistics of the population.

        Parameters:
        -----------
        population : Population
            The population containing multiple portfolios.

        Returns:
        --------
        pd.DataFrame
            The summary statistics DataFrame.
        """
        try:
            summary = population.summary()
            st.subheader("Annualized Sharpe Ratio")
            st.dataframe(summary.loc["Annualized Sharpe Ratio"].to_frame().T)

            # Full Summary
            st.subheader("Full Summary Statistics")
            st.dataframe(summary)

            logger.info("Displayed summary statistics for the population.")
            return summary
        except Exception as e:
            logger.error(f"Error displaying summary statistics: {e}")
            return None

# %%
# Streamlit App Layout
# =====================
def main():
    st.set_page_config(page_title="HRP, HERC & NCO Portfolio Optimization with Economic Factors", layout="wide")
    st.title("Hierarchical Risk Parity (HRP), Hierarchical Equal Risk Contribution (HERC) & Nested Clusters Optimization (NCO) Portfolio Optimization with Economic Factors")

    # Initialize session state for storing data
    if 'models' not in st.session_state:
        st.session_state['models'] = []
    if 'weights_df' not in st.session_state:
        st.session_state['weights_df'] = pd.DataFrame()

    # Sidebar for Configuration
    st.sidebar.header("Configuration")

    # File Uploads
    with st.sidebar.expander("Upload Data Files", expanded=True):
        market_file = st.file_uploader("Market Index Data File (Excel/CSV)", type=["xlsx", "xls", "csv"])
        risk_free_rate_file = st.file_uploader("Risk-Free Rate Data File (Excel/CSV)", type=["xlsx", "xls", "csv"])
        market_cap_file = st.file_uploader("Market Cap in USD Data File (Excel/CSV)", type=["xlsx", "xls", "csv"])
        usd_to_rial_file = st.file_uploader("USD to Rial Exchange Rate Data File (Excel/CSV)", type=["xlsx", "xls", "csv"])

    # Symbol IDs Input
    with st.sidebar.expander("Symbol IDs", expanded=True):
        symbol_ids_input = st.text_area(
            "Enter Symbol IDs separated by commas",
            value="17914401175772326,66682662312253625,28374437855144739",
            help="Input the symbol IDs you wish to analyze, separated by commas."
        )
        portfolio_symbol_ids = [id_.strip() for id_ in symbol_ids_input.split(",")]

    # Model Parameters
    with st.sidebar.expander("Model Parameters", expanded=True):
        linkage_methods = st.multiselect(
            "Linkage Methods",
            options=["average", "complete", "single", "ward"],
            default=["average", "complete"],
            help="Select the linkage methods to use in hierarchical clustering."
        )
        distance_estimators = st.multiselect(
            "Distance Estimators",
            options=["Pearson", "Kendall", "Spearman", "DistanceCorrelation"],  # افزودن "DistanceCorrelation"
            default=["Pearson", "Kendall", "Spearman"],
            help="Select the distance estimators to use."
        )
        risk_aversion = st.slider("Risk Aversion", 0.5, 2.0, 1.0, step=0.1)
        cvar_beta = st.slider("CVaR Beta", 0.90, 0.99, 0.95, step=0.01)
        wasserstein_ball_radius = st.slider("Wasserstein Ball Radius", 0.01, 0.05, 0.02, step=0.01)
        
        # افزودن تنظیمات مربوط به NCO در رابط کاربری
        st.markdown("### Nested Clusters Optimization (NCO) Parameters")
        nco_inner_estimator = st.selectbox(
            "NCO Inner Estimator",
            options=["MeanRisk", "HierarchicalRiskParity", "DistributionallyRobustCVaR"],
            index=0,
            help="Select the inner estimator for NCO."
        )
        nco_outer_estimator = st.selectbox(
            "NCO Outer Estimator",
            options=["MeanRisk", "HierarchicalRiskParity", "DistributionallyRobustCVaR"],
            index=0,
            help="Select the outer estimator for NCO."
        )
        nco_clustering_method = st.selectbox(
            "NCO Clustering Method",
            options=["HierarchicalClustering", "KMeans"],
            index=0,
            help="Select the clustering method for NCO."
        )
        nco_cv = st.selectbox(
            "NCO Cross-Validation Strategy",
            options=["ignore", "5-fold", "10-fold"],
            index=0,
            help="Select the cross-validation strategy for NCO."
        )
        nco_quantile = st.slider("NCO Quantile", 0.0, 1.0, 0.5, step=0.05)
        nco_quantile_measure = st.selectbox(
            "NCO Quantile Measure",
            options=["Sharpe Ratio", "Sortino Ratio"],  # حذف "Maximum Drawdown"
            index=0,
            help="Select the quantile measure for NCO."
        )
        nco_n_jobs = st.number_input(
            "NCO Number of Jobs",
            min_value=-1,
            max_value=32,
            value=-1,
            step=1,
            help="Number of jobs to run in parallel for NCO. -1 means using all processors."
        )

        # افزودن تنظیمات مربوط به SpearmanDistance
        st.markdown("### Spearman Distance Parameters")
        spearman_absolute = st.checkbox(
            "Apply Absolute Transformation",
            value=False,
            help="If checked, the absolute transformation is applied to the Spearman correlation matrix."
        )
        spearman_power = st.slider(
            "Spearman Distance Power",
            0.1, 3.0, 1.0, step=0.1,
            help="Exponent of the power transformation applied to the Spearman correlation matrix."
        )

        # افزودن تنظیمات مربوط به DistanceCorrelation
        st.markdown("### Distance Correlation Parameters")
        distance_correlation_threshold = st.slider(
            "Distance Correlation Threshold",
            0.1, 1.0, 0.5, step=0.1,
            help="Distance correlation threshold."
        )

    # Start Button
    start_button = st.sidebar.button("Run HRP, HERC & NCO Optimization")

    # ==============================
    # Data Upload and Validation
    # ==============================
    st.header("1. Upload Data Files")
    data_files = {}
    if market_file and risk_free_rate_file and market_cap_file and usd_to_rial_file:
        data_files = {
            'market': DataLoader.load_data('market', market_file),
            'risk_free_rate': DataLoader.load_data('risk_free_rate', risk_free_rate_file),
            'market_cap': DataLoader.load_data('market_cap', market_cap_file),
            'usd_to_rial': DataLoader.load_data('usd_to_rial', usd_to_rial_file)
        }

        # Check if all data files are loaded successfully
        if all(not df.empty for df in data_files.values()):
            st.success("All data files uploaded and processed successfully!")
            # Display previews
            st.subheader("Data Previews")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Market Index Data**")
                st.dataframe(data_files['market'].head())
            with col2:
                st.markdown("**Risk-Free Rate Data**")
                st.dataframe(data_files['risk_free_rate'].head())

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Market Cap Data**")
                st.dataframe(data_files['market_cap'].head())
            with col4:
                st.markdown("**USD to Rial Exchange Rate Data**")
                st.dataframe(data_files['usd_to_rial'].head())
        else:
            st.error("One or more data files failed to load. Please check the uploaded files.")
    else:
        st.warning("Please upload all required data files.")

    # ==============================
    # Symbol IDs Validation
    # ==============================
    st.header("2. Symbol IDs")
    if portfolio_symbol_ids:
        if validate_symbol_ids(portfolio_symbol_ids):
            st.success(f"Symbol IDs: {', '.join(portfolio_symbol_ids)}")
        else:
            st.error("Invalid symbol IDs provided. Ensure all IDs are numeric.")
    else:
        st.error("Please enter at least one symbol ID.")

    # ==============================
    # Run HRP, HERC & NCO Optimization
    # ==============================
    if start_button:
        if not all(not df.empty for df in data_files.values()):
            st.error("Please ensure all data files are uploaded and processed successfully.")
        elif not validate_symbol_ids(portfolio_symbol_ids):
            st.error("Please provide valid numeric symbol IDs.")
        else:
            # Proceed with data fetching and preprocessing
            with st.spinner("Fetching symbol data..."):
                symbol_data = asyncio.run(DataFetcher.fetch_all_symbols(portfolio_symbol_ids))

            # Combine all symbol data into a single DataFrame
            prices = pd.DataFrame()
            for symbol_id, df in symbol_data.items():
                if not df.empty:
                    df = df[['date', 'close']].rename(columns={'close': symbol_id})
                    if prices.empty:
                        prices = df
                    else:
                        prices = prices.merge(df, on='date', how='inner')  # Use 'inner' to keep only overlapping dates

            if prices.empty:
                st.error("No price data fetched for the provided symbol IDs. Please check the symbol IDs and try again.")
            else:
                prices = prices.set_index('date').sort_index()
                st.success(f"Successfully fetched and combined price data. Total data points: {len(prices)}")
                st.subheader("Price Data Preview")
                st.dataframe(prices.head())

                # Extract Market Returns, Market Cap, USD to Rial, and Risk-Free Rate
                try:
                    market_returns = data_files['market'].set_index('date')['daily_return'].rename("market_returns")
                    risk_free_rate = data_files['risk_free_rate'].set_index('date')['risk_free_rate'].rename("risk_free_rate")
                    market_cap = data_files['market_cap'].set_index('date')['market_cap'].rename("market_cap")
                    usd_to_rial = data_files['usd_to_rial'].set_index('date')['usd_to_rial'].rename("usd_to_rial")
                except KeyError as e:
                    st.error(f"Missing required column in data files: {e}")
                    st.stop()

                # Data Preprocessing
                with st.spinner("Preprocessing data..."):
                    preprocessor = Preprocessor()
                    X_train, X_test, y_train, y_test = preprocessor.process_data(
                        prices, market_returns, risk_free_rate, market_cap, usd_to_rial
                    )

                if X_train is None:
                    st.error("Data preprocessing failed. Please check your data and try again.")
                else:
                    st.success("Data preprocessing completed successfully!")
                    st.subheader("Training Data Preview")
                    st.dataframe(X_train.head())

                    st.subheader("Testing Data Preview")
                    st.dataframe(X_test.head())

                    # ==============================
                    # Model Configuration and Training
                    # ==============================
                    st.header("3. Model Training and Evaluation")
                    evaluator = Evaluator()

                    # Define the linkage methods and distance estimators from user input
                    linkage_methods_enum = [LinkageMethod(method) for method in linkage_methods]
                    distance_estimators_list = []
                    for estimator_name in distance_estimators:
                        if estimator_name == "Pearson":
                            distance_estimators_list.append((None, "Pearson"))
                        elif estimator_name == "Kendall":
                            distance_estimators_list.append((KendallDistance(absolute=True), "Kendall"))
                        elif estimator_name == "Spearman":
                            # دریافت تنظیمات SpearmanDistance از رابط کاربری
                            spearman = SpearmanDistance(
                                absolute=spearman_absolute,
                                power=spearman_power
                            )
                            distance_estimators_list.append((spearman, "Spearman"))
                        elif estimator_name == "DistanceCorrelation":
                            # دریافت تنظیمات DistanceCorrelation از رابط کاربری
                            distance_corr = DistanceCorrelation(
                                threshold=distance_correlation_threshold
                            )
                            distance_estimators_list.append((distance_corr, "DistanceCorrelation"))
                        else:
                            st.error(f"Unsupported distance estimator: {estimator_name}")
                            st.stop()

                    # Initialize an empty list to hold all models
                    models = []

                    # Iterate over each combination of linkage methods and distance estimators for HRP
                    for linkage in linkage_methods_enum:
                        for distance_estimator, distance_name in distance_estimators_list:
                            # Define a name for the model based on the linkage and distance method
                            distance_label = distance_name if distance_estimator else "Pearson"
                            model_name = f"HRP-{linkage.value.capitalize()}-{distance_label}"

                            # Initialize HierarchicalClustering with the current linkage method
                            hierarchical_clustering = HierarchicalClustering(linkage_method=linkage)

                            # Initialize HierarchicalRiskParity optimizer with the clustering estimator and distance estimator
                            optimizer = HierarchicalRiskParity(
                                risk_measure=RiskMeasure.CVAR,
                                distance_estimator=distance_estimator,
                                prior_estimator=None,  # No prior used in standard HRP
                                hierarchical_clustering_estimator=hierarchical_clustering,
                                portfolio_params=dict(name=model_name)
                            )

                            # Create an OptimizerModel instance and add it to the models list
                            model = OptimizerModel(optimizer=optimizer, name=model_name)
                            models.append(model)

                    # ========================================
                    # اضافه کردن مدل HierarchicalEqualRiskContribution (HERC)
                    # ========================================

                    # تعریف و آموزش FactorModel
                    factor_model = FactorModel()
                    factor_model.fit(X_train, y_train)
                    logger.info("FactorModel has been fitted.")

                    # تعریف مدل HierarchicalEqualRiskContribution (HERC)
                    optimizer_HERC = HierarchicalEqualRiskContribution(
                        risk_measure=RiskMeasure.VARIANCE,  # می‌توانید این را بر اساس نیاز تغییر دهید
                        prior_estimator=factor_model,
                        distance_estimator=None,  # به طور پیش‌فرض PearsonDistance
                        hierarchical_clustering_estimator=HierarchicalClustering(linkage_method=LinkageMethod.WARD),
                        min_weights=0.0,  # پورتفوی بدون فروش کوتاه
                        max_weights=1.0,  # هر دارایی حداکثر 100%
                        solver='CLARABEL',
                        solver_params=None,
                        transaction_costs=0.0,
                        management_fees=0.0,
                        previous_weights=None,
                        portfolio_params=dict(name="HERC-FactorModel")
                    )

                    # ایجاد و اضافه کردن مدل HERC به لیست مدل‌ها
                    model_HERC = OptimizerModel(optimizer=optimizer_HERC, name="HERC-FactorModel")
                    models.append(model_HERC)
                    logger.info("HERC-FactorModel has been added to the models list.")

                    # ========================================
                    # اضافه کردن مدل NestedClustersOptimization (NCO)
                    # ========================================

                    # تعریف Inner Estimator برای NCO
                    if nco_inner_estimator == "MeanRisk":
                        inner_optimizer = EqualWeighted(portfolio_params=dict(name="NCO-Inner-MeanRisk"))
                    elif nco_inner_estimator == "HierarchicalRiskParity":
                        inner_optimizer = HierarchicalRiskParity(
                            risk_measure=RiskMeasure.CVAR,
                            distance_estimator=None,
                            prior_estimator=None,
                            hierarchical_clustering_estimator=HierarchicalClustering(linkage_method=LinkageMethod.AVERAGE),
                            portfolio_params=dict(name="NCO-Inner-HRP")
                        )
                    elif nco_inner_estimator == "DistributionallyRobustCVaR":
                        inner_optimizer = DistributionallyRobustCVaR(
                            risk_aversion=1.0,
                            cvar_beta=0.95,
                            wasserstein_ball_radius=0.02,
                            prior_estimator=factor_model,
                            min_weights=0.0,
                            max_weights=1.0,
                            budget=1.0,
                            portfolio_params=dict(name="NCO-Inner-DRCVaR"),
                            solver='CLARABEL',
                            solver_params=None,
                            scale_objective=None,
                            scale_constraints=None,
                            save_problem=False,
                            raise_on_failure=True
                        )
                    else:
                        st.error(f"Unsupported Inner Estimator for NCO: {nco_inner_estimator}")
                        st.stop()

                    # تعریف Outer Estimator برای NCO
                    if nco_outer_estimator == "MeanRisk":
                        outer_optimizer = EqualWeighted(portfolio_params=dict(name="NCO-Outer-MeanRisk"))
                    elif nco_outer_estimator == "HierarchicalRiskParity":
                        outer_optimizer = HierarchicalRiskParity(
                            risk_measure=RiskMeasure.CVAR,
                            distance_estimator=None,
                            prior_estimator=None,
                            hierarchical_clustering_estimator=HierarchicalClustering(linkage_method=LinkageMethod.AVERAGE),
                            portfolio_params=dict(name="NCO-Outer-HRP")
                        )
                    elif nco_outer_estimator == "DistributionallyRobustCVaR":
                        outer_optimizer = DistributionallyRobustCVaR(
                            risk_aversion=1.0,
                            cvar_beta=0.95,
                            wasserstein_ball_radius=0.02,
                            prior_estimator=factor_model,
                            min_weights=0.0,
                            max_weights=1.0,
                            budget=1.0,
                            portfolio_params=dict(name="NCO-Outer-DRCVaR"),
                            solver='CLARABEL',
                            solver_params=None,
                            scale_objective=None,
                            scale_constraints=None,
                            save_problem=False,
                            raise_on_failure=True
                        )
                    else:
                        st.error(f"Unsupported Outer Estimator for NCO: {nco_outer_estimator}")
                        st.stop()

                    # تعریف Clustering Estimator برای NCO
                    if nco_clustering_method == "HierarchicalClustering":
                        clustering_estimator = HierarchicalClustering(linkage_method=LinkageMethod.WARD)
                    elif nco_clustering_method == "KMeans":
                        from sklearn.cluster import KMeans
                        clustering_estimator = KMeans(n_clusters=5, random_state=42)
                    else:
                        st.error(f"Unsupported Clustering Method for NCO: {nco_clustering_method}")
                        st.stop()

                    # تعریف Cross-Validation Strategy برای NCO
                    if nco_cv == "ignore":
                        cv_strategy = "ignore"
                    elif nco_cv == "5-fold":
                        from sklearn.model_selection import KFold
                        cv_strategy = KFold(n_splits=5, shuffle=False)
                    elif nco_cv == "10-fold":
                        from sklearn.model_selection import KFold
                        cv_strategy = KFold(n_splits=10, shuffle=False)
                    else:
                        st.error(f"Unsupported CV Strategy for NCO: {nco_cv}")
                        st.stop()

                    # تعریف Quantile Measure برای NCO
                    quantile_measure_map = {
                        "Sharpe Ratio": RatioMeasure.SHARPE_RATIO,  # اصلاح شده از RiskMeasure به RatioMeasure
                        "Sortino Ratio": RatioMeasure.SORTINO_RATIO
                    }
                    quantile_measure = quantile_measure_map.get(nco_quantile_measure, RatioMeasure.SHARPE_RATIO)

                    # تعریف مدل NCO
                    optimizer_NCO = NestedClustersOptimization(
                        inner_estimator=inner_optimizer,
                        outer_estimator=outer_optimizer,
                        distance_estimator=None,  # به طور پیش‌فرض PearsonDistance
                        clustering_estimator=clustering_estimator,
                        cv=cv_strategy,
                        quantile=nco_quantile,
                        quantile_measure=quantile_measure,
                        n_jobs=int(nco_n_jobs),
                        verbose=0,
                        portfolio_params=dict(name="NCO-FactorModel")
                    )

                    # ایجاد و اضافه کردن مدل NCO به لیست مدل‌ها
                    model_NCO = OptimizerModel(optimizer=optimizer_NCO, name="NCO-FactorModel")
                    models.append(model_NCO)
                    logger.info("NCO-FactorModel has been added to the models list.")

                    # تعریف مدل DistributionallyRobustCVaR با FactorModel به عنوان prior_estimator
                    optimizer_drcvar = DistributionallyRobustCVaR(
                        risk_aversion=risk_aversion,
                        cvar_beta=cvar_beta,
                        wasserstein_ball_radius=wasserstein_ball_radius,
                        prior_estimator=factor_model,
                        min_weights=0.0,
                        max_weights=1.0,
                        budget=1.0,
                        portfolio_params=dict(name="DistributionallyRobustCVaR-Factor-Model"),
                        solver='CLARABEL',
                        solver_params=None,
                        scale_objective=None,
                        scale_constraints=None,
                        save_problem=False,
                        raise_on_failure=True
                    )
                    model_drcvar = OptimizerModel(optimizer=optimizer_drcvar, name="DistributionallyRobustCVaR-Factor-Model")
                    models.append(model_drcvar)

                    # Fit Models
                    with st.spinner("Fitting models..."):
                        for model in models:
                            # مدل‌هایی که در نامشان 'Factor' وجود دارد، از y_train استفاده می‌کنند
                            if "Factor" in model.name:
                                model.fit(X_train, y_train)
                            else:
                                model.fit(X_train)

                    st.success("All models have been fitted successfully!")

                    # Benchmarking with Equal Weighted Estimator
                    bench_optimizer = EqualWeighted(portfolio_params=dict(name="EqualWeighted"))
                    benchmark = OptimizerModel(optimizer=bench_optimizer, name="EqualWeighted")
                    benchmark.fit(X_train)
                    models.append(benchmark)

                    # Risk Contribution Analysis and Dendrograms
                    st.subheader("Risk Contribution Analysis and Dendrograms")
                    for model in models[:-1]:  # Exclude benchmark
                        st.markdown(f"### Model: {model.name}")
                        portfolio = model.predict(X_train)
                        if portfolio is not None:
                            evaluator.analyze_risk_contribution(portfolio, measure=model.risk_measure)
                            model.plot_dendrogram(heatmap=True)
                        else:
                            st.warning(f"Could not generate risk contribution for model '{model.name}'.")

                    # ==============================
                    # Prediction on Test Set and Evaluation
                    # ==============================
                    st.header("4. Portfolio Prediction and Evaluation")
                    population_test = Population([])
                    for model in models:
                        portfolio = model.predict(X_test)
                        if portfolio is not None:
                            population_test.append(portfolio)
                            logger.info(f"Appended portfolio from model '{model.name}' to population.")
                        else:
                            logger.warning(f"Skipped appending portfolio from model '{model.name}' due to prediction failure.")

                    # Plot Cumulative Returns
                    st.subheader("Cumulative Returns")
                    evaluator.plot_cumulative_returns(population_test)

                    # Plot Portfolio Composition
                    st.subheader("Portfolio Composition")
                    evaluator.plot_composition(population_test)

                    # Print Summary Statistics and Retrieve Summary DataFrame
                    st.subheader("Summary Statistics")
                    summary = evaluator.print_summary(population_test)

                    if summary is None:
                        st.error("Summary statistics could not be generated.")
                    else:
                        # ==============================
                        # Export Optimized Portfolios and Rank Them
                        # ==============================
                        st.header("5. Export Optimized Portfolios")

                        try:
                            # Initialize an empty dictionary to store weights
                            portfolio_weights = {}

                            # Iterate over each model and extract weights
                            for model in models:
                                # Predict portfolio on the training set to get the weights
                                portfolio = model.predict(X_train)
                                if portfolio is not None:
                                    # Ensure weights are a Series with symbol IDs as index
                                    weights = pd.Series(portfolio.weights, index=X_train.columns)
                                    portfolio_weights[model.name] = weights
                                    logger.info(f"Extracted weights for model '{model.name}'.")
                                else:
                                    logger.warning(f"Cannot extract weights for model '{model.name}' as prediction failed.")

                            # Create a DataFrame from the weights dictionary
                            weights_df = pd.DataFrame(portfolio_weights).transpose()

                            # Reset index to have model names as a column
                            weights_df = weights_df.reset_index().rename(columns={'index': 'Model'})

                            # Ranking based on Annualized Sharpe Ratio
                            sharpe_ratios = summary.loc["Annualized Sharpe Ratio"]
                            ranking_df = sharpe_ratios.reset_index().rename(columns={'index': 'Model', 'Annualized Sharpe Ratio': 'Sharpe Ratio'})
                            ranking_df['Rank'] = ranking_df['Sharpe Ratio'].rank(ascending=False, method='dense').astype(int)
                            ranking_df = ranking_df.sort_values('Rank')

                            # Merge ranking into weights_df
                            weights_df = weights_df.merge(ranking_df[['Model', 'Sharpe Ratio', 'Rank']], on='Model', how='left')

                            # Reorder columns to have 'Rank' first
                            weights_df = weights_df[['Rank', 'Model'] + [col for col in weights_df.columns if col not in ['Rank', 'Model']]]

                            # Display the weights DataFrame
                            st.subheader("Optimized Portfolio Weights")
                            st.dataframe(weights_df)

                            # Display the ranking
                            st.subheader("Portfolio Rankings based on Annualized Sharpe Ratio")
                            st.dataframe(ranking_df.sort_values('Rank'))

                            # Export to CSV and Excel with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            csv = weights_df.to_csv(index=False)

                            # Create Excel bytes
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                weights_df.to_excel(writer, index=False)
                            excel_data = output.getvalue()

                            st.markdown(get_table_download_link(weights_df, f'Optimized_Portfolio_Weights_{timestamp}.csv', '📥 Download CSV'), unsafe_allow_html=True)
                            st.markdown(get_table_download_link(weights_df, f'Optimized_Portfolio_Weights_{timestamp}.xlsx', '📥 Download Excel'), unsafe_allow_html=True)

                            logger.info(f"Exported optimized portfolio weights to 'Optimized_Portfolio_Weights_{timestamp}.csv' and '.xlsx'.")
                        except Exception as e:
                            logger.error(f"Error exporting portfolio weights: {e}")
                            st.error("An error occurred while exporting portfolio weights.")

# %%
# Execute Main Function
# ======================
if __name__ == "__main__":
    main()
