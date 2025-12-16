import gradio as gr
import data_loader
import book_recommender




def create_dashboard():

    loader = data_loader.BookDataLoader("books.csv")
    df = loader.load_and_preprocess()

    with gr.Blocks() as demo:
        gr.Markdown("# Book Recommendation System")

        with gr.Tab("Content-Based Recommendation"):
            gr.Markdown("### Content-Based Recommendation: Recommendations based on Genre similarity, Author, Page Count, and Ratings")
            with gr.Row():
                with gr.Column():
                    book_title = gr.Textbox(label="Book Title", max_lines=2)
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column():
                    output = gr.Textbox(label="Recommendations")
        with gr.Tab("Popularity-Based Recommendation"):
            gr.Markdown("### Popularity-Based Recommendation: Recommendations based on Popularity, with the option to filter by Genre")
            with gr.Row():
                with gr.Column():
                    book_title = gr.Textbox(label="Book Title", max_lines=2)
                    genre = gr.Textbox(label="Genre (Optional)")
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column():
                    output = gr.Textbox(label="Recommendations")
        with gr.Tab("Hybrid-Based Recommendations"):
            gr.Markdown("### Hybrid-Based Recommendation: Recommendations based on combining content-based and popularity Recommendation Systems ")
            with gr.Row():
                with gr.Column():
                    book_title = gr.Textbox(label="Book Title", max_lines=2)
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column():
                    output = gr.Textbox(label="Recommendations")



    
    return demo

if __name__ == '__main__':
    demo = create_dashboard()
    demo.launch(theme=gr.themes.Soft())