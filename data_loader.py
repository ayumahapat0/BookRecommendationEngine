"""
Data Loader Module for Book Recommendation Engine
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import ast


class BookDataLoader:
    """
    Efficiently loads and preprocesses the Goodreads book dataset.

    Demonstrates Week 12 concepts:
    - Memory-efficient data loading with usecols
    - dtype optimization
    - Chunked processing (optional for very large datasets)
    """

    # TODO: Define the columns you want to load
    # Hint: You don't need ALL columns. Think about what's essential for recommendations!
    COLUMNS_TO_LOAD = [
        'book_id',
        'book_title',
        'author',
        'genres',
        'num_ratings',
        'num_reviews',
        'average_rating',
        'num_pages'
    ]

    # TODO: Specify optimal dtypes for memory efficiency
    # Hint: Consider the range of values and whether you need float64 or if float32 suffices
    DTYPE_SPECIFICATION = {
        'book_id': 'int32',  # IDs don't need int64
        'book_title': 'string',  # Use pandas string type
        'author': 'string',
        'num_ratings': 'int32',  # Rating counts fit in int32
        'num_reviews': 'int32',
        'average_rating': 'float32',  # float32 is sufficient for ratings
        'num_pages': 'string'  # Will need parsing (stored as list)
    }

    def __init__(self, filepath: str):
        """
        Initialize the data loader.

        Args:
            filepath: Path to the CSV file
        """
        self.filepath = filepath
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset with optimal memory usage.

        TODO: Implement efficient data loading
        - Use usecols to load only necessary columns
        - Specify dtypes to reduce memory usage
        - Handle the index column properly

        Returns:
            DataFrame with loaded data
        """
        # TODO: Load data efficiently
        # Hint: Use pd.read_csv with usecols and dtype parameters
        self.data = pd.read_csv(
            self.filepath,
            usecols=self.COLUMNS_TO_LOAD,
            dtype=self.DTYPE_SPECIFICATION,
        )

        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the loaded data.

        TODO: Implement preprocessing steps
        1. Parse the 'genres' column (it's a string representation of a list)
        2. Parse the 'num_pages' column (also a list, take the first element)
        3. Handle missing values appropriately
        4. Create any derived features that might be useful

        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # TODO: Parse genres from string to list
        # Hint: Use ast.literal_eval() to safely evaluate string representations of lists
        self.data['genres'] = self.data['genres'].apply(self._parse_genres)

        # TODO: Parse num_pages and convert to integer
        self.data['num_pages'] = self.data['num_pages'].apply(self._parse_pages)

        # TODO: Handle missing values
        # Consider: Should you drop rows or fill with defaults?
        self.data = self.data.dropna(subset=['book_title', 'author', 'average_rating'])

        # Fill missing genres with empty list
        self.data['genres'] = self.data['genres'].apply(lambda x: x if isinstance(x, list) else [])

        # Fill missing pages with median
        median_pages = self.data['num_pages'].median()
        self.data['num_pages'].fillna(median_pages, inplace=True)

        # TODO: Create a 'popularity_score' feature
        # Hint: Combine num_ratings and average_rating in a meaningful way
        # Think about Week 10: What's an efficient way to compute this?
        self.data['popularity_score'] = self._calculate_popularity()

        return self.data

    def _parse_genres(self, genre_str: str) -> List[str]:
        """
        Parse genres from string representation to list.

        TODO: Implement genre parsing
        Handle cases where the string might be malformed or empty

        Args:
            genre_str: String representation of genre list

        Returns:
            List of genres
        """
        try:
            # TODO: Safely parse the string
            genres = ast.literal_eval(genre_str)
            return genres if isinstance(genres, list) else []
        except (ValueError, SyntaxError):
            return []

    def _parse_pages(self, pages_str: str) -> float:
        """
        Parse number of pages from string representation.

        TODO: Implement page number parsing
        The format is like "['652']" - extract the number

        Args:
            pages_str: String representation of pages list

        Returns:
            Number of pages as float (or np.nan if invalid)
        """
        try:
            # TODO: Parse and convert to integer
            pages_list = ast.literal_eval(pages_str)
            if isinstance(pages_list, list) and len(pages_list) > 0:
                if pages_list[0] == None:
                    return np.nan
                else:
                    return float(pages_list[0])
        except (ValueError, SyntaxError):
            pass
        return np.nan

    def _calculate_popularity(self) -> pd.Series:
        """
        Calculate a popularity score for each book.

        TODO: Design a popularity metric
        Consider both the number of ratings and the average rating

        Week 10 concept: Use vectorized operations for efficiency!

        Returns:
            Series of popularity scores
        """
        # TODO: Implement popularity calculation
        # Example approach: normalized_ratings * average_rating
        # Normalize num_ratings to 0-1 scale first

        max_ratings = self.data['num_ratings'].max()
        normalized_ratings = self.data['num_ratings'] / max_ratings

        # Weight both factors
        popularity = normalized_ratings * (self.data['average_rating'] / 5.0)

        return popularity

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Report memory usage statistics.

        This demonstrates the impact of dtype optimization (Week 12 concept)

        Returns:
            Dictionary with memory usage information
        """
        if self.data is None:
            return {"error": "No data loaded"}

        memory_usage = self.data.memory_usage(deep=True)

        return {
            "total_memory_mb": memory_usage.sum() / 1024 ** 2,
            "per_column_mb": (memory_usage / 1024 ** 2).to_dict(),
            "shape": self.data.shape
        }

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Convenience method to load and preprocess in one call.

        Returns:
            Preprocessed DataFrame ready for recommendation engine
        """
        self.load_data()
        return self.preprocess_data()


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = BookDataLoader("books.csv")

    print("Loading data...")
    df = loader.load_and_preprocess()

    print("\n=== Data Info ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    print("\n=== Memory Usage ===")
    memory_info = loader.get_memory_usage()
    print(f"Total Memory: {memory_info['total_memory_mb']:.2f} MB")

    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Data Types ===")
    print(df.dtypes)