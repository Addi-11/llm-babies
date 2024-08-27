import streamlit as st
import pickle
import tiktoken
import matplotlib.pyplot as plt
from regex_tokenizer import RegexTokenizer
from evaluation import TokenizerEvaluation

st.set_page_config(layout="wide")

# Define pastel colors
pastel_colors = [
    "#E0F2FE", "#FEF3C7", "#BFDBFE", "#D1FAE5", "#FBD38D", "#C4F1F9", "#E5E7EB", "#D6BCFA", "#C4B5FD", "#E2F5D8", "#FECACA", "#D6BCFA", "#F9E2B1", "#D1FAE5", "#E4E4E7", "#FEE2E2",  "#F5A8F2", "#FBCFE8", "#A7F3D0"
]

st.title("Hindi Tokenizer Visualizer")

def load_tokenizer(filename):
    """Load and return the tokenizer from a pickle file."""
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def create_token_dict(encoded_tokens, tokenizer):
    decoded_words = [tokenizer.decode([token]) for token in encoded_tokens]
    token_dictionary = {token: word for token, word in zip(encoded_tokens, decoded_words)}

    return token_dictionary

def highlight(items):
    """Generate HTML for highlighted text with pastel colors."""
    highlighted_text = ""
    i = 0
    for item in items:
        color = pastel_colors[i % len(pastel_colors)]
        highlighted_text += f"<span style='background-color: {color}; padding: 2px;'>{item}</span> "
        i += 1
    return highlighted_text

def render_html_section(title, html_content):
    """Render an HTML section with the given title and content."""
    st.markdown(f"""
        <h3>{title}</h3>
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 4px; background-color: #f9f9f9;'>
            {html_content}
        </div>
    """, unsafe_allow_html=True)

def display_evaluation_metrics(evaluator, text):
    """Display evaluation metrics for the given text."""
    metrics = {
        "Vocabulary Size": evaluator.vocabulary_size(),
        "Fertility Score": evaluator.fertility_score(text),
        "Token Coverage": evaluator.token_coverage(text),
        "Subword Count": evaluator.subword_count(text),
        "Compression Ratio": evaluator.compression_ratio(text),
        "Perplexity": evaluator.perplexity(text),
        "Consistency": evaluator.consistency([text]),
        "Tokenization Speed": evaluator.tokenization_speed(text),
        "Entropy": evaluator.entropy(text),
        "Character Coverage": evaluator.character_coverage(text),
    }

    st.subheader("Evaluation Metrics")
    for metric, value in metrics.items():
        st.write(f"**{metric}**: {value}")

    # Token length distribution
    token_lengths, bin_edges = evaluator.token_length_distribution(text)
    st.subheader("Token Length Distribution")
    fig, ax = plt.subplots()
    ax.bar(bin_edges[:-1], token_lengths, width=bin_edges[1] - bin_edges[0], color='skyblue')
    ax.set_xlabel('Token Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Length Distribution')
    st.pyplot(fig)

def main():
    tokenizer = load_tokenizer('hindi_tokenizer/hindi_tokenizer.pkl')
    # tokenizer = load_tokenizer('reg_tokenizer/reg_tokenizer_11550steps.pkl')
    # tokenizer = tiktoken.get_encoding("o200k_base")
    evaluator = TokenizerEvaluation(tokenizer)
    

    col1, col2 = st.columns([1.5, 1])

    with col1:
        text = st.text_area("Enter text to tokenize", "नमस्ते दुनिया!")
        if st.button("Tokenize"):
        
            encoded_tokens = tokenizer.encode(text)
            token_dict = create_token_dict(encoded_tokens, tokenizer)

            highlighted_text = highlight(token_dict.values())
            render_html_section("Highlighted Text", highlighted_text)

            token_list = highlight(token_dict.keys())
            render_html_section("Corresponding Tokens", token_list)

            with col2:
                display_evaluation_metrics(evaluator, text)


if __name__ == "__main__":
    main()
