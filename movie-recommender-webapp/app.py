import streamlit as st
import requests
import pickle


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={moviedb_key}&language=en-US".format(movie_id)
    print(f"Fetching poster from URL: {url}")  
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

   
    return recommended_movie_names,recommended_movie_posters

def main():

    st.markdown("<h1 style='text-align: center; color: black;'>Movie Recommender System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Find a similar movie from a dataset of 5,000 movies!</h4>", unsafe_allow_html=True)

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select the movie you like :",
        movie_list
    )

    if st.button('Show Recommendation'):
        st.write("Recommended movies based on your interest are :")
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        cols= st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(recommended_movie_names[i])
                st.image(recommended_movie_posters[i])

    st.title(" ")



if __name__ == "__main__":
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    main()