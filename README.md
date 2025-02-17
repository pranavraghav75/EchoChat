# 🌟 EchoAI: Celebrity Chat with Personalized AI

EchoAI is an AI-powered chat application that lets you have dynamic, in-depth conversations with simulated versions of your favorite celebrities. By aggregating and analyzing data from multiple sources—such as YouTube transcripts, Wikipedia, TMZ, Reddit, and Google News—EchoAI crafts a detailed personality profile and speaking style guide for each celebrity. The result is a conversational experience where the AI authentically embodies the celebrity’s unique voice and mannerisms.

---

## 🦅 Features

- **Celebrity Persona Emulation:** Engage with a simulated version of a celebrity using their real-world data.
- **Multi-Source Data Aggregation:** Gathers information from Wikipedia, TMZ, Reddit, Google News, and YouTube to build comprehensive profiles.
- **Personality Analysis:** Leverages NLP, clustering, and summarization techniques to extract and replicate speech patterns.
- **Conversational AI:** Powered by OpenAI's GPT-4o-mini model, ensuring responses reflect the celebrity’s distinct personality.
- **User-Friendly Interface:** Streamlit-based frontend provides an intuitive and responsive chat experience.
- **API-Driven Architecture:** Built with FastAPI for scalable and efficient backend processing.

---

## 💻 Tech Stack

- **FastAPI:** Backend API framework for handling endpoints and processing.
- **Streamlit:** For the UI and user interactions.
- **ChromaDB:** Persistent vector database for storing contextual conversation data.
- **OpenAI GPT-4o-mini:** AI engine for generating authentic responses and personality emulation.
- **NLTK & Scikit-Learn:** Natural language processing, text tokenization, and clustering for personality analysis.
- **BeautifulSoup & Requests:** Data scraping from TMZ, Wikipedia, and other web sources.
- **youtube_transcript_api:** For fetching YouTube video transcripts.
- **asyncpraw:** Asynchronous Reddit API wrapper for data retrieval.

---

## 🚀 Running the Application

From the project root, run:

```bash
uvicorn api:app --reload
```

The FastAPI server will start on [http://localhost:8000](http://localhost:8000).

### Launch the Streamlit Frontend

In a new terminal (with the virtual environment activated), run:

```bash
streamlit run streamlit_app.py
```

This opens the EchoChat interface in your default web browser.

---

## 🤖 How It Works

1. **Session Initialization:**  
   - Enter a celebrity’s name to start a chat session.
   - The API aggregates data from Wikipedia, TMZ, Reddit, Google News, and YouTube.
   - A personality profile and detailed speaking style guide are generated through clustering and summarization techniques on transcipts from their interviews public on YouTube.

2. **Conversational Chat:**  
   - The `/chat` endpoint uses conversation history, contextual data, and the personality profile to generate authentic responses in the celebrity's style.
   - Responses are powered by OpenAI's GPT-4o-mini model and refined through custom prompts.

3. **Session Cleanup:**  
   - End your chat session to remove stored conversation data and free up resources.

---

## 🔌 API Endpoints Overview

- **GET /**  
  _Welcome endpoint for basic connectivity check._

- **GET /wikipedia**  
  _Fetches summary data from Wikipedia for a given celebrity._

- **POST /init_session**  
  _Initializes a chat session by aggregating data and generating a personality profile._

- **POST /chat**  
  _Processes user messages and returns AI-generated responses in the celebrity's style._

- **POST /cleanup_session**  
  _Cleans up session data after the chat session ends._

---

## 🙏 Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/).
- Data sourced from [Wikipedia](https://www.wikipedia.org/), [TMZ](https://www.tmz.com/), [Reddit](https://www.reddit.com/), [Google News](https://news.google.com/), and [YouTube](https://www.youtube.com/).
- Personality analysis powered by OpenAI's GPT-4o-mini model.

---

Happy coding and enjoy chatting with your favorite celebrities with EchoAI!
