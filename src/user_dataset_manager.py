import pandas as pd
import os

class UserDatasetManager:
    """Handles loading, analyzing, and summarizing user-provided datasets."""

    def __init__(self, note_taker=None):
        self.note_taker = note_taker

    def load_dataset(self, file_path):
        """
        Load a dataset from a file path (CSV or Excel).
        Returns a pandas DataFrame or None if loading fails.
        """
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found at '{file_path}'")
            return None
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                print(f"❌ Error: Unsupported file format. Please use CSV or Excel.")
                return None
            
            print(f"✅ Dataset '{os.path.basename(file_path)}' loaded successfully.")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def sanitize_columns(self, df: pd.DataFrame):
        """
        Sanitizes column names to be valid Python identifiers.

        - Converts to lowercase
        - Replaces spaces and special characters with underscores
        - Removes leading numbers or adds a prefix

        Returns the sanitized DataFrame and a mapping from new names to old names.
        """
        import re
        sanitized_columns = {}
        original_to_new = {}

        for col in df.columns:
            new_col = col.strip()
            new_col = new_col.lower()
            new_col = re.sub(r'[\s\W-]+', '_', new_col) # Replace whitespace and non-alphanumeric with _
            new_col = re.sub(r'^(\d)', r'_\1', new_col) # Add _ prefix if starts with a digit
            new_col = re.sub(r'_+', '_', new_col) # Collapse multiple underscores
            new_col = new_col.strip('_') # Remove leading/trailing underscores

            # Handle potential duplicates
            original_new_col = new_col
            counter = 1
            while new_col in sanitized_columns.values():
                new_col = f"{original_new_col}_{counter}"
                counter += 1

            sanitized_columns[col] = new_col
            original_to_new[col] = new_col

        df.columns = df.columns.map(sanitized_columns)
        
        # Create a mapping from new names back to original names for reporting
        new_to_original_mapping = {v: k for k, v in sanitized_columns.items()}

        print("   ✅ Column names sanitized for safe access.")
        return df, new_to_original_mapping, original_to_new

    def analyze_dataset(self, df, target_variable=None, column_name_mapping=None):
        """
        Performs a deep analysis of the dataset and returns a structured dictionary.
        """
        if df is None:
            return None

        analysis = {}
        
        # Basic metadata
        analysis['shape'] = df.shape
        analysis['columns'] = df.columns.tolist()
        analysis['column_name_mapping'] = column_name_mapping

        # Target variable info
        analysis['target_info'] = {
            'original_target_variable': self._get_original_name(target_variable, column_name_mapping),
            'target_variable': target_variable,
            'task_type': self._infer_task_type(df, target_variable) if target_variable else 'unknown'
        }
        
        # Missing values analysis
        missing_total = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        analysis['missing_percentage'] = (missing_total / total_cells) * 100 if total_cells > 0 else 0
        analysis['missing_per_column'] = {col: df[col].isnull().sum() for col in df.columns}

        # Dataset size warning
        min_rows_for_training = 100
        if df.shape[0] < min_rows_for_training:
            analysis['size_warning'] = f"Warning: The dataset has {df.shape[0]} rows, which may be too small for reliable model training (recommended > {min_rows_for_training})."
        else:
            analysis['size_warning'] = None

        # Class balance analysis (if target variable is provided)
        if target_variable:
            if target_variable not in df.columns:
                analysis['class_balance_error'] = f"Error: Target variable '{target_variable}' not found in the dataset columns."
                analysis['class_balance'] = None
            else:
                balance = df[target_variable].value_counts(normalize=True).to_dict()
                analysis['class_balance'] = {str(k): v * 100 for k, v in balance.items()} # percentage
                analysis['class_balance_error'] = None
        else:
            analysis['class_balance'] = "Not assessed (no target variable provided)."
            analysis['class_balance_error'] = None
            
        return analysis

    def _get_original_name(self, sanitized_name, mapping):
        """Helper to get original column name from sanitized name."""
        if not mapping or not sanitized_name:
            return sanitized_name
        return mapping.get(sanitized_name, sanitized_name)

    def _infer_task_type(self, df, target_variable):
        """Infers if the task is classification or regression."""
        if target_variable not in df.columns:
            return 'unknown'
        
        target_series = df[target_variable]
        
        # If the column is numeric and has high cardinality, assume regression.
        if pd.api.types.is_numeric_dtype(target_series):
            if target_series.nunique() > 25: # Heuristic for regression
                return 'regression'
        
        return 'classification'

    def get_summary_text(self, analysis):
        """
        Generates a human-readable summary of the dataset analysis.
        """
        if not analysis:
            return "No analysis available."

        summary = []
        summary.append(f"Rows: {analysis['shape'][0]}, Columns: {analysis['shape'][1]}")
        summary.append(f"Columns: {', '.join(analysis['columns'])}")
        
        if analysis['size_warning']:
            summary.append(f"📢 {analysis['size_warning']}")
            
        summary.append(f"Missing Values: {analysis['missing_percentage']:.2f}% of total cells.")
        
        if isinstance(analysis['class_balance'], dict):
            summary.append("Class Balance:")
            for class_label, percentage in analysis['class_balance'].items():
                summary.append(f"  - {class_label}: {percentage:.2f}%")
        elif analysis['class_balance_error']:
            summary.append(f"⚠️  Class Balance: {analysis['class_balance_error']}")
        else:
            summary.append(f"Class Balance: {analysis['class_balance']}")
            
        return "\n".join(summary)

    def log_dataset_details(self, analysis_results):
        """Logs the detailed analysis to the note taker."""
        if self.note_taker and analysis_results:
            self.note_taker.log(
                event_type="dataset_analysis",
                data=analysis_results
            ) 