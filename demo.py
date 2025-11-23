"""
Demo Script for Book Recommendation Engine
"""

import time
import pandas as pd
from data_loader import BookDataLoader
from book_recommender import BookRecommendationEngine


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demonstrate_data_loading():
    """
    Demonstrate efficient data loading with memory optimization.

    TODO:
    1. Load data twice - once with all columns, once with optimized columns
    2. Compare memory usage
    3. Show the impact of dtype optimization
    """
    print_section("PART 1: EFFICIENT DATA LOADING")

    # TODO: Load data with optimization
    print("\n📥 Loading data with optimized columns and dtypes...")
    start_time = time.time()

    loader = BookDataLoader("books.csv")
    books_df = loader.load_and_preprocess()

    load_time = time.time() - start_time

    print(f"✅ Data loaded in {load_time:.2f} seconds")
    print(f"📊 Dataset shape: {books_df.shape}")

    # TODO: Show memory usage
    print("\n💾 Memory Usage Analysis:")
    memory_info = loader.get_memory_usage()
    print(f"Total memory: {memory_info['total_memory_mb']:.2f} MB")

    print("\nMemory per column:")
    for col, mem in memory_info['per_column_mb'].items():
        if isinstance(mem, float):
            print(f"  {col:20s}: {mem:.2f} MB")

    # TODO: Show data types
    print("\n📋 Optimized Data Types:")
    print(books_df.dtypes)

    # TODO: Show sample data
    print("\n👀 Sample Data (first 3 rows):")
    print(books_df.head(3)[['book_title', 'author', 'average_rating', 'num_ratings']])

    return books_df

def get_unique_genres(genres):
    """
    Get all the unqiue genres from the dataset

    Args: 
        genre = list of genres associated with each novel
    Returns:
        res = list of all the unique genres 
    """
    res = [x for genre in genres for x in genre]
    return res

def demonstrate_memory_optimization():
    """
        Demonstrate how much memory optimizations is done using dtype optimizations
    """
    loader = BookDataLoader("books.csv")
    
    data_normal = loader.load_data_normal()
    data_dtype_opt = loader.load_data()

    memory_before = data_normal.memory_usage(deep=True).sum() / 1024 ** 2
    memory_after = data_dtype_opt.memory_usage(deep=True).sum() / 1024 ** 2

    improvement = ((memory_before - memory_after) / memory_before) * 100

    print("Memory Usage Before Dtype Optimizations")
    print("Before:")
    print(f"Total Memory Used: {memory_before:2f} MB\n")
    print("After:")
    print(f"Total Memory Used: {memory_after:2f} MB\n")
    print(f"Percentage Improvement: {improvement:2f}%\n")


def demonstrate_content_based(engine: BookRecommendationEngine):
    """
    Demonstrate content-based recommendations.

    TODO: Test with multiple books and show results
    """
    print_section("PART 2: CONTENT-BASED RECOMMENDATIONS")

    test_books = [
        "1984",
        "The Great Gatsby"
    ]

    for book_title in test_books:
        print(f"\n🔍 Finding books similar to: '{book_title}'")
        print("-" * 80)

        # TODO: Get content-based recommendations
        start_time = time.time()
        recommendations = engine.get_recommendations(
            book_title,
            strategy='content',
            n=5
        )
        elapsed = time.time() - start_time

        if recommendations:
            engine.display_recommendations(recommendations)
            print(f"\n⏱️  Recommendation time: {elapsed:.4f} seconds")
        else:
            print(f"❌ Book not found in dataset")


def demonstrate_popularity(engine: BookRecommendationEngine, genre):
    """
    Demonstrate popularity-based recommendations.

    TODO: Show top popular books overall and by genre
    """
    print_section("PART 3: POPULARITY-BASED RECOMMENDATIONS")

    print(f"\n📈 Top 5 Most Popular Books in Genres: {genre}")
    print("-" * 80)

    # TODO: Get popularity recommendations
    
    # For Popularity Recommendations, we don't need a title 
    test_book = ""
    start_time = time.time()
    popular_books = engine.get_recommendations(test_book, strategy='popularity', genre=genre, n=5)
    elapsed = time.time() - start_time

    if popular_books:
        engine.display_recommendations(popular_books)
        print(f"\n⏱️  Recommendation time: {elapsed:.4f} seconds")


def demonstrate_hybrid(engine: BookRecommendationEngine):
    """
    Demonstrate hybrid recommendations.

    TODO: Show how hybrid combines content and popularity
    """
    print_section("PART 4: HYBRID RECOMMENDATIONS")

    test_book = "Harry Potter and the Half-Blood Prince"

    print(f"\n🎯 Hybrid recommendations for: '{test_book}'")
    print("(Combining content similarity + popularity)")
    print("-" * 80)

    # TODO: Get hybrid recommendations
    start_time = time.time()
    recommendations = engine.get_recommendations(
        test_book,
        strategy='hybrid',
        n=5
    )
    elapsed = time.time() - start_time

    if recommendations:
        engine.display_recommendations(recommendations)
        print(f"\n⏱️  Recommendation time: {elapsed:.4f} seconds")


def compare_strategies(engine: BookRecommendationEngine):
    """
    Compare all three recommendation strategies side by side.

    TODO: Show how different strategies produce different results
    """
    print_section("PART 5: STRATEGY COMPARISON")

    test_book = "The Hunger Games"

    print(f"\n📊 Comparing recommendation strategies for: '{test_book}'")
    print("=" * 80)

    strategies = ['content','popularity', 'hybrid']
    results = {}

    for strategy in strategies:
        print(f"\n🔸 Strategy: {strategy.upper()}")
        print("-" * 80)

        start_time = time.time()
        recommendations = engine.get_recommendations(test_book, strategy=strategy, n=5)
        elapsed = time.time() - start_time

        results[strategy] = recommendations

        if recommendations:
            # Show just titles and scores for comparison
            for i, (book_id, title, score) in enumerate(recommendations, 1):
                print(f"{i}. {title[:50]:50s} | Score: {score:.3f}")
            print(f"\n⏱️  Time: {elapsed:.4f}s")
        else:
            print("❌ No recommendations found")


    # TODO: Analyze differences
    print("\n📝 Analysis:")
    print("-" * 80)
    print("Content-based focuses on similar features (genres, author, etc.)")
    print("Popularity-based focuses solely on popularity score, not on anything else")
    print("Hybrid balances similarity with overall popularity")
    print("\nNotice how the recommendations and scores differ between strategies!")


def performance_analysis(engine: BookRecommendationEngine):
    """
    Analyze performance characteristics of the recommendation engine.

    TODO: Time multiple operations and analyze scalability
    """
    print_section("PART 6: PERFORMANCE ANALYSIS")

    print("\n⚡ Timing Analysis:")
    print("-" * 80)

    test_books = [
        "Harry Potter",
        "1984",
        "The Hobbit"
    ]

    times = []

    for book in test_books:
        start = time.time()
        recommendations = engine.get_recommendations(book, strategy='content', n=10)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"Recommendations for '{book:30s}': {elapsed:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"\n📊 Average recommendation time: {avg_time:.4f}s")

    print("\n💡 Performance Notes:")
    print("- Similarity matrix is pre-computed (one-time cost)")
    print("- Each recommendation query is fast (just lookups and sorts)")
    print("- For larger datasets, consider on-demand similarity computation")


def main():
    """
    Main demonstration script.

    TODO: Complete all demonstration sections
    """
    print("\n" + "🎬" * 40)
    print("  BOOK RECOMMENDATION ENGINE DEMONSTRATION")
    print("  CS 5130 - Lab 6")
    print("🎬" * 40)

    # Part 1: Data Loading
    books_df = demonstrate_data_loading()
    genres = books_df['genres'].tolist()

    unique_genres = get_unique_genres(genres)

    print_section("DEMONSTRATING EFFECT OF DTYPE OPTIMIZATIONS")
    demonstrate_memory_optimization()

    # Build recommendation engine
    print_section("BUILDING RECOMMENDATION ENGINE")
    print("\n🔧 Initializing recommendation engine...")
    print("   (This may take a moment while computing similarity matrix...)")

    start_time = time.time()
    engine = BookRecommendationEngine(books_df)
    build_time = time.time() - start_time

    print(f"✅ Engine ready! (built in {build_time:.2f} seconds)")

    # Part 2-5: Different recommendation strategies
    demonstrate_content_based(engine)
    demonstrate_popularity(engine, unique_genres[0])
    demonstrate_hybrid(engine)
    compare_strategies(engine)

    # Part 6: Performance analysis
    performance_analysis(engine)

    # Conclusion
    print_section("DEMONSTRATION COMPLETE")
    print("\n✨ All recommendation strategies demonstrated successfully!")
    print("\n💡 Key Takeaways:")
    print("   1. Efficient data loading saves significant memory")
    print("   2. Different strategies have different strengths")
    print("   3. Vectorized operations enable fast recommendations")
    print("   4. Good software design makes the system extensible")

    print("\n" + "=" * 80)
    print("Thank you for using the Book Recommendation Engine! 📚")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()