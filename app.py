import gradio as gr
import data_loader
import book_recommender




def create_dashboard():

    loader = data_loader.BookDataLoader("books.csv")
    df = loader.load_and_preprocess()
    engine = book_recommender.BookRecommendationEngine(df)
    genres = data_loader.BookDataLoader.get_unique_genres(df['genres'].tolist())
    if len(genres) > 100:
        genres = genres[:100]


    def content_based_recommendations(book_title, num_books):
        """
        Provides Content-based Recommendations

        Args: 
            book_title: book title
            num_books: number of books to recommend
        Returns:
            String containing recommendations
        """
        
        if len(book_title) < 1 or book_title.isspace():
            return "Must Enter Valid Title String"
        recommendations = engine.get_recommendations(book_title, 'content', num_books)

        if recommendations and len(recommendations) > 0:
            return engine.display_recommendations(recommendations)
        else:
            return "No Recommendations"
    
    def popularity_recommendations(genre, num_books):
        """
        Provides Popularity-based Recommendations

        Args: 
            genre: genre
            num_books: number of books to recommend
        Returns:
            String containing recommendations
        """
        
        recommendations = engine.get_recommendations("", 'popularity', genre=genre, n=num_books)

        if recommendations and len(recommendations) > 0:
            return engine.display_recommendations(recommendations)
        else:
            return "No Recommendations"
    
    def hybrid_recommendations(book_title, num_books):
        """
        Provides Hybrid-based Recommendations

        Args: 
            book_title: book title
            num_books: number of books to recommend
        Returns:
            String containing recommendations
        """
        
        if len(book_title) < 1 or book_title.isspace():
            return "Must Enter Valid Title String"
        recommendations = engine.get_recommendations(book_title, 'hybrid', num_books)

        if recommendations and len(recommendations) > 0:
            return engine.display_recommendations(recommendations)
        else:
            return "No Recommendations"

    with gr.Blocks() as demo:
        gr.Markdown("# Book Recommendation System")

        with gr.Tab("Content-Based Recommendation"):
            gr.Markdown("### Content-Based Recommendation: Recommendations based on Genre similarity, Author, Page Count, and Ratings")
            
            with gr.Row():
                
                with gr.Column():
                    book_title = gr.Textbox(label="Book Title", max_lines=2)
                    num_books = gr.Slider(minimum=1, maximum=25, value=5, step=1, interactive=True)
                    rec_button = gr.Button("Get Recommendations")
                
                with gr.Column():
                    output = gr.Textbox(label="Recommendations", lines=25)
                
                rec_button.click(
                    fn=content_based_recommendations,
                    inputs=[book_title, num_books],
                    outputs=[output] 
                )
        with gr.Tab("Popularity-Based Recommendation"):
            gr.Markdown("### Popularity-Based Recommendation: Recommendations based on Popularity, with the option to filter by Genre")
            
            with gr.Row():
               
                with gr.Column():
                    genre = gr.Dropdown(label="Genre",choices=genres, value=None, interactive=True)
                    num_books = gr.Slider(minimum=1, maximum=25, value=5, step=1, interactive=True)
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column():
                    output = gr.Textbox(label="Recommendations", lines=25)
                
                rec_button.click(
                    fn=popularity_recommendations,
                    inputs=[genre, num_books],
                    outputs=[output] 
                )

        with gr.Tab("Hybrid-Based Recommendations"):
            gr.Markdown("### Hybrid-Based Recommendation: Recommendations based on combining content-based and popularity Recommendation Systems ")
            
            with gr.Row():
                
                with gr.Column():
                    book_title = gr.Textbox(label="Book Title", max_lines=2)
                    num_books = gr.Slider(minimum=1, maximum=25, value=5, step=1, interactive=True)
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column():
                    output = gr.Textbox(label="Recommendations", lines=25)
                
                rec_button.click(
                    fn=hybrid_recommendations,
                    inputs=[book_title, num_books],
                    outputs=[output] 
                )



    
    return demo

if __name__ == '__main__':
    demo = create_dashboard()
    demo.launch(theme=gr.themes.Soft())