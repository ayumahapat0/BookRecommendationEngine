"""
Book Recommendation Engine
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class RecommendationStrategy:
    """
    Base class for recommendation strategies (Week 8: OOP Design)

    This demonstrates the Strategy pattern - different algorithms
    can be swapped in and out easily.
    """

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Generate recommendations for a given book.

        Args:
            book_id: The book to base recommendations on
            n: Number of recommendations to return

        Returns:
            List of tuples: (book_id, title, score)
        """
        raise NotImplementedError("Subclasses must implement recommend()")
    
    def recommend(self, genre: str, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Generates recommendations for books based on popularity score

        Args:
            genre: optional filter to find popularity within genre
            n: NUmber of recommendations to return
        
        Returns:
            List of tuples: (book_id, book_title, popularity_score)   
        """
        raise NotImplementedError("Subclasses must implement recommend()")



class ContentBasedRecommender(RecommendationStrategy):
    """
    Content-based recommendation using book features using:
    - Genre similarity
    - Author matching
    - Other book features
    """

    def __init__(self, books_df: pd.DataFrame):
        """
        Initialize the recommender with book data.

        Args:
            books_df: DataFrame with book information
        """
        self.books_df = books_df.copy()
        self.similarity_matrix = None
        self._prepare_features()

    def _prepare_features(self):
        """
        Prepare feature vectors for similarity computation.
        - One-hot encode genres (use MultiLabelBinarizer)
        - Consider author, page count, ratings
        """

        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(self.books_df['genres'])

        normalized_pages = (
                self.books_df['num_pages'] / self.books_df['num_pages'].max()
        )
        normalized_rating = self.books_df['average_rating'] / 5.0

        # Combine features
        # Shape: (n_books, n_genre_features + 2)
        self.feature_matrix = np.hstack([
            genre_features,
            normalized_pages.values.reshape(-1, 1),
            normalized_rating.values.reshape(-1, 1)
        ])

        self._compute_similarity_matrix()

    def _compute_similarity_matrix(self):
        """
        Compute pairwise similarity between all books.
        - Using scipy/sklearn optimized implementations
        - For very large datasets, computing on-demand might be better
        """

        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Recommend books similar to the given book.

        Args:
            book_id: ID of the book to base recommendations on
            n: Number of recommendations

        Returns:
            List of (book_id, title, similarity_score)
        """

        try:
            idx = self.books_df[self.books_df['book_id'] == book_id].index[0]
        except IndexError:
            return []


        sim_scores = self.similarity_matrix[idx]

        similar_indices = np.argsort(sim_scores)[::-1][1:n + 1]

        recommendations = []
        for idx in similar_indices:
            book_row = self.books_df.iloc[idx]
            recommendations.append((
                book_row['book_id'],
                book_row['book_title'],
                float(sim_scores[idx])
            ))

        return recommendations


class PopularityRecommender(RecommendationStrategy):
    """
    Recommend popular books, optionally filtered by genre.

    This is simpler but serves as a good baseline.
    """

    def __init__(self, books_df: pd.DataFrame):
        self.books_df = books_df.copy()

    def recommend(self, genre: str = None, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Recommend top books by popularity.

        Args:
            genre: Optional genre filter
            n: Number of recommendations

        Returns:
            List of (book_id, title, popularity_score)
        """
        df = self.books_df.copy()

        if genre:
            df = df[df['genres'].apply(lambda x: genre in x)]

        top_books = df.nlargest(n, 'popularity_score')

        recommendations = []
        for _, row in top_books.iterrows():
            recommendations.append((
                row['book_id'],
                row['book_title'],
                row['popularity_score']
            ))

        return recommendations


class HybridRecommender(RecommendationStrategy):
    """
    Combines multiple recommendation strategies:
    - Content-based similarity
    - Popularity

    Week 7: Use functional programming concepts to combine strategies
    """

    def __init__(self, books_df: pd.DataFrame,
                 content_weight: float = 0.7,
                 popularity_weight: float = 0.3):
        """
        Initialize hybrid recommender.

        Args:
            books_df: Book data
            content_weight: Weight for content-based score
            popularity_weight: Weight for popularity score
        """
        self.content_recommender = ContentBasedRecommender(books_df)
        self.popularity_recommender = PopularityRecommender(books_df)
        self.books_df = books_df
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Generate hybrid recommendations.

        Args:
            book_id: Book to base recommendations on
            n: Number of recommendations

        Returns:
            List of (book_id, title, combined_score)
        """
        content_recs = self.content_recommender.recommend(book_id, n * 3)

        content_scores = {book_id: score for book_id, _, score in content_recs}

        candidate_ids = [book_id for book_id, _, _ in content_recs]
        candidate_books = self.books_df[self.books_df['book_id'].isin(candidate_ids)]

        max_pop = candidate_books['popularity_score'].max()

        combined_scores = []
        for _, row in candidate_books.iterrows():
            book_id = row['book_id']
            content_score = content_scores.get(book_id, 0)
            popularity_score = row['popularity_score'] / max_pop if max_pop > 0 else 0

            # Weighted combination
            combined = (self.content_weight * content_score +
                        self.popularity_weight * popularity_score)

            combined_scores.append((
                book_id,
                row['book_title'],
                combined
            ))

        combined_scores.sort(key=lambda x: x[2], reverse=True)
        return combined_scores[:n]


class BookRecommendationEngine:
    """
    Main interface for the recommendation system.

    Week 9: Clean, maintainable design with clear responsibilities
    """

    def __init__(self, books_df: pd.DataFrame):
        """
        Initialize the recommendation engine.

        Args:
            books_df: Preprocessed book data
        """
        self.books_df = books_df
        self.strategies = {
            'content': ContentBasedRecommender(books_df),
            'popularity': PopularityRecommender(books_df),
            'hybrid': HybridRecommender(books_df)
        }

    def get_recommendations(self,
                            book_title: str,
                            strategy: str = 'hybrid', 
                            genre: str = None,
                            n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Get book recommendations.

        Args:
            book_title: Title of book to base recommendations on
            strategy: 'content', 'popularity', or 'hybrid'
            genre: genre of book to filter for popularity recommendations
            n: Number of recommendations

        Returns:
            List of recommended books with scores
        """
        # Popuular Strategy doesn't used book titles, only finds books based on popularity score, with optional genre filter
        if strategy == 'popularity':
            return self.strategies[strategy].recommend(genre, n)
        
        # Find book by title
        matches = self.books_df[
            self.books_df['book_title'].str.contains(book_title, case=False, na=False)
        ]

        if matches.empty:
            print(f"Book '{book_title}' not found")
            return []
        

        book_id = matches.iloc[0]['book_id']

        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        else:
            return self.strategies[strategy].recommend(book_id, n)

    def display_recommendations(self, recommendations: List[Tuple[int, str, float]]):
        """
        Pretty print recommendations.

        Args:
            recommendations: List of (book_id, title, score) tuples
        """
        res = ""

        for i, (book_id, title, score) in enumerate(recommendations, 1):
            # Get book details
            book = self.books_df[self.books_df['book_id'] == book_id].iloc[0]

            res += f"{i}. {title}"
            res += f"   Author: {book['author']}"
            res += f"   Rating: {book['average_rating']:.2f} ⭐ ({book['num_ratings']:,} ratings)"
            res += f"   Genres: {', '.join(book['genres'][:3])}"
            res += f"   Match Score: {score:.3f}\n\n"

        return res


# Example usage
if __name__ == "__main__":
    from data_loader import BookDataLoader

    # Load data
    print("Loading data...")
    loader = BookDataLoader("goodreads_books_2024.csv")
    books_df = loader.load_and_preprocess()

    # Create recommendation engine
    print("Building recommendation engine...")
    engine = BookRecommendationEngine(books_df)

    # Test recommendations
    test_book = "Harry Potter"
    print(f"\nGetting recommendations for books like '{test_book}'...")

    for strategy in ['content', 'popularity', 'hybrid']:
        print(f"\n{'=' * 80}")
        print(f"Strategy: {strategy.upper()}")
        print('=' * 80)

        recommendations = engine.get_recommendations(test_book, strategy=strategy, n=5)
        engine.display_recommendations(recommendations)