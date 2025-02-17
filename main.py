import os
import uuid
import openai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import tweepy
import asyncpraw
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from collections import defaultdict

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
chroma_path = os.getenv("CHROMADB_URL")
google_news_api_key = os.getenv("GOOGLE_NEWS_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
hf_token = os.getenv("HF_TOKEN")
client = OpenAI(api_key=openai_api_key)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-ada-002"
)

chroma_client = chromadb.PersistentClient(path=chroma_path)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

# twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)

async def fetch_reddit_data(query, num_results=10):
    reddit = asyncpraw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent
    )
    subreddit = await reddit.subreddit("all")
    results = []

    async for submission in subreddit.search(query, limit=num_results):
        post_info = {
            "title": submission.title,
            "selftext": submission.selftext,
        }
        results.append(post_info)

    return results if results else ["No Reddit posts found."]

async def fetch_tmz_data(celebrity: str, max_results: int = 30):
    try:
        query = celebrity.replace(" ", "+")
        url = f'https://www.tmz.com/search/?q={query}/'
        response = requests.get(url)
        if response.status_code != 200:
            return f"Error fetching TMZ data: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('a', {'class': 'gridler__card-link gridler__card-link--default js-track-link js-click-article'})
        result_texts = []

        for article in articles[:max_results]:
            title = article.get_text(strip=True)
            link = article['href']
            result_texts.append(f"Title: {title}\nLink: https://www.tmz.com{link}\n")
        return "\n".join(result_texts) if result_texts else "No articles found."
    except Exception as e:
        return f"Failed to fetch TMZ articles: {e}"

async def fetch_google_news_data(query: str, page_size: int = 30) -> str:
    try:
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={google_news_api_key}&pageSize={page_size}'
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            news_data = [article['title'] + ": " + article['description'] for article in articles]
            return "\n".join(news_data) if news_data else "No news articles found."
        else:
            return "Failed to retrieve Google News data."
    except Exception as e:
        return "Failed to retrieve Google News data."

def fetch_wikipedia_data(celebrity: str) -> dict:
    try:
        formatted_name = celebrity.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_name}"
        response = requests.get(url)
        if response.status_code != 200:
            return {}
        data = response.json()

        return {
            "title": data.get("title", ""),
            "summary": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "image": data.get("thumbnail", {}).get("source", "")
        }
    except Exception as e:
        return {}

def fetch_youtube_videos(celebrity_name, max_results=5):
    try:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
        search_query = f"{celebrity_name} speech or interview"
        search_response = youtube.search().list(
            q=search_query,
            part="id",
            maxResults=max_results,
            type="video",
            videoDuration="medium"
        ).execute()
        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
        
        if not video_ids:
            return []
        return video_ids
    except Exception as e:
        return []

def cluster_speech_segments(transcript_data, n_clusters=2):
    sentences = []
    timestamps = []
    for segment in transcript_data:
        segment_sentences = sent_tokenize(segment['text'])
        sentences.extend(segment_sentences)
        timestamps.extend([{'start': segment['start'], 'duration': segment['duration']}] * len(segment_sentences))
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )

    if len(sentences) <= n_clusters:
        return [{'start': t['start'], 'end': t['start'] + t['duration'], 'speaker': f'SPEAKER_00', 'text': s} for s, t in zip(sentences, timestamps)]
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(tfidf_matrix)
        speaker_profiles = defaultdict(list)

        for sentence, cluster_id in zip(sentences, clusters):
            speaker_profiles[f'SPEAKER_{cluster_id:02d}'].append(sentence)
        clustered_segments = []

        for sentence, timestamp, cluster_id in zip(sentences, timestamps, clusters):
            clustered_segments.append({
                'start': timestamp['start'],
                'end': timestamp['start'] + timestamp['duration'],
                'speaker': f'SPEAKER_{cluster_id:02d}',
                'text': sentence
            })
        smoothed_segments = smooth_speaker_transitions(clustered_segments)
        return smoothed_segments
    
    except Exception as e:
        print(f"Clustering failed: {str(e)}")
        return [{'start': t['start'], 'end': t['start'] + t['duration'], 'speaker': 'SPEAKER_00', 'text': s} for s, t in zip(sentences, timestamps)]

def smooth_speaker_transitions(segments, window_size=3):
    if len(segments) <= window_size:
        return segments
    smoothed = segments.copy()

    for i in range(window_size, len(segments) - window_size):
        window = segments[i-window_size:i+window_size+1]
        speakers = [s['speaker'] for s in window]
        majority_speaker = max(set(speakers), key=speakers.count)
        smoothed[i]['speaker'] = majority_speaker
    return smoothed

def estimate_speaker_count(transcript_data):
    total_text = ' '.join([segment['text'] for segment in transcript_data])
    sentences = sent_tokenize(total_text)

    if len(sentences) < 10:
        return 2
    elif len(sentences) < 30:
        return 3
    else:
        return 4

def filter_celebrity_speech(clusters, celebrity_name):
    celebrity_clusters = []
    for cluster in clusters:
        excerpt = cluster['text'][:300]
        prompt = f"Determine if the following transcript excerpt is spoken by {celebrity_name} (the celebrity) or by an interviewer.\n\nExcerpt: \"{excerpt}\"\n\nAnswer only with 'celebrity' or 'interviewer'."
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )

            classification = response.choices[0].message.content.strip().lower()
            if 'celebrity' in classification:
                celebrity_clusters.append(cluster)

        except Exception as e:
            print(f"Error classifying cluster: {e}")
            continue
    
    return celebrity_clusters

def fetch_youtube_transcripts(video_ids, celebrity_name):
    results = []
    for video_id in video_ids:
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'])
            n_clusters = estimate_speaker_count(transcript_data)
            clustered_segments = cluster_speech_segments(transcript_data, n_clusters)
            speaker_texts = defaultdict(list)

            for segment in clustered_segments:
                speaker_texts[segment['speaker']].append(segment['text'])
            clusters = []

            for speaker, texts in speaker_texts.items():
                clusters.append({
                    'speaker': speaker,
                    'text': ' '.join(texts)
                })
            celebrity_clusters = filter_celebrity_speech(clusters, celebrity_name)

            for cluster in celebrity_clusters:
                results.append({
                    'video_id': video_id,
                    'speaker': cluster['speaker'],
                    'text': cluster['text']
                })
        except Exception as e:
            continue

    return results

def extract_quotes(interviews, target_speaker=None):
    quotes = []
    for interview in interviews:
        if isinstance(interview, dict):
            speaker = interview.get('speaker')
            text = interview.get('text', '')
            if target_speaker and speaker != target_speaker:
                continue
            text = text.replace("\n", " ")
            sentences = sent_tokenize(text)
            for sentence in sentences:
                quotes.append({
                    'speaker': speaker,
                    'quote': sentence.strip()
                })

    return quotes

def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    word_freq = Counter(words)

    ranked_sentences = sorted(sentences, key=lambda s: sum(word_freq[w] for w in word_tokenize(s.lower())), reverse=True)
    return " ".join(ranked_sentences[:num_sentences])

def abstractive_summary(text):
    prompt = f"Summarize the following news while keeping the most important details:\n\n{text}"
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def fetch_personality(celebrity_name, interviews):
    quotes = extract_quotes(interviews)
    if not quotes:
        return f"Not enough data to analyze {celebrity_name}'s speech style."
    
    quote_text = "\n".join([f"Speaker {q['speaker']}: {q['quote']}" for q in quotes])
    prompt = f"""You are an expert linguistic analyst specializing in personality and speech pattern modeling. Analyze {celebrity_name}'s speaking style from these interview quotes:

{quote_text}

Create a comprehensive speaking style guide that covers:

1. VOICE CHARACTERISTICS:
- Tempo (fast/slow, rhythm patterns)
- Energy level and enthusiasm
- Tone variations and emphasis patterns
- Signature vocal mannerisms

2. LANGUAGE PATTERNS:
- Vocabulary level and preferences
- Catchphrases or repeated expressions
- Sentence structure (simple/complex, direct/roundabout)
- Use of slang, humor, or profanity
- Metaphors or analogies they favor

3. CONVERSATIONAL STYLE:
- How they handle questions (direct/evasive, thoughtful/quick)
- Story-telling techniques
- How they express opinions
- Their way of agreeing/disagreeing
- Use of personal anecdotes

4. PERSONALITY MARKERS:
- Emotional expressiveness
- Confidence level in responses
- How they handle controversial topics
- Self-deprecation or boasting tendencies
- Treatment of others in conversation

5. CONTEXTUAL VARIATIONS:
- How their style changes between casual/formal settings
- Response patterns to criticism or praise
- Their go-to deflection or emphasis strategies

Present this analysis in a way that allows precise replication of their speaking style. Include specific examples from the quotes where possible.
"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

@app.get("/")
def read_root():
    return {"message": "Welcome to the EchoAI API!"}

@app.get("/wikipedia")
def get_wikipedia(celebrity: str):
    data = fetch_wikipedia_data(celebrity)
    if not data or not data.get("title"):
        raise HTTPException(status_code=404, detail="Wikipedia page not found.")
    return data

@app.post("/init_session")
async def init_session(celebrity: str):
    session_id = str(uuid.uuid4())
    celebrity = celebrity.title()
    video_ids = fetch_youtube_videos(celebrity)

    interviews = fetch_youtube_transcripts(video_ids, celebrity)
    personality = fetch_personality(celebrity, interviews)

    tmz_data = await fetch_tmz_data(celebrity)
    reddit_data = await fetch_reddit_data(celebrity)
    news_data = await fetch_google_news_data(celebrity)
    wiki_data = fetch_wikipedia_data(celebrity)

    tmz_summary = extractive_summary(tmz_data)
    reddit_text = "\n".join([f"Title: {post['title']}\nText: {post['selftext']}" for post in reddit_data])
    reddit_summary = extractive_summary(reddit_text)
    news_summary = abstractive_summary(news_data)
    wiki_summary = abstractive_summary(wiki_data) or wiki_data.get('summary', 'No summary available.')

    combined_info = (
        f"From TMZ: {tmz_summary}\n"
        f"From Reddit: {reddit_summary}\n"
        f"From Google News: {news_summary}\n"
        f"From Wikipedia: {wiki_summary}"
    )

    try:
        collection = chroma_client.create_collection(
            name=session_id, embedding_function=openai_ef
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create session collection.")
    embedding = openai_ef([combined_info])[0]
    collection.upsert(ids=["context_data"], documents=[combined_info], embeddings=[embedding])
    sessions[session_id] = {
        "celebrity": celebrity,
        "collection": collection,
        "conversation": [],
        "personality": personality
    }
    return {"session_id": session_id, "data": combined_info, "personality": personality}

@app.post("/chat")
async def chat(session_id: str, user_message: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_data = sessions[session_id]
    collection = session_data["collection"]
    personality = session_data["personality"]

    try:
        results = collection.query(query_texts=[user_message], n_results=3)
    except Exception as e:
        results = {"documents": [[""]]}
    
    retrieved_context = " ".join([doc for docs in results.get("documents", []) for doc in docs if doc])
    conversation_history = "\n".join([f"User: {conv['user']}\nBot: {conv['bot']}" for conv in session_data["conversation"]])
    system_prompt = f"""You are {session_data['celebrity']}. Embody their personality completely using this detailed speaking style guide:

{personality}

IMPORTANT GUIDELINES:
1. Stay in character 100% of the time - never break the fourth wall
2. Use the exact speech patterns, vocabulary, and mannerisms described in the style guide
3. Reference real events and relationships from your life naturally
4. React to topics using your known views and opinions
5. Include your characteristic emotional expressions and vocal patterns
6. Use your typical humor style where appropriate
7. Reference your personal experiences and relationships organically
8. Match your documented confidence level and way of handling different types of questions
9. Incorporate your signature phrases and expressions naturally, not forcefully
10. Adjust your tone based on the formality/casualness of the question
11. Make it conversational and sounding human, don't spit out tons of info for a simple question
12. If the celebrity doesn't like someone or something, express it in a subtle way

CONTEXTUAL INFORMATION:
{retrieved_context}

PREVIOUS CONVERSATION:
{conversation_history}

Respond to the user's message exactly as you would in a real interview or conversation, maintaining complete authenticity to your personality and speaking style.
"""
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150,
        temperature=0.8,
    )

    answer = response.choices[0].message.content.strip()
    session_data["conversation"].append({"user": user_message, "bot": answer})
    interaction_text = f"User: {user_message}\nBot: {answer}"
    interaction_embedding = openai_ef([interaction_text])[0]
    collection.upsert(
        ids=[str(uuid.uuid4())],
        documents=[interaction_text],
        embeddings=[interaction_embedding]
    )
    return {"answer": answer}

@app.post("/cleanup_session")
async def cleanup_session(session_id: str):
    if session_id in sessions:
        try:
            chroma_client.delete_collection(name=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to delete session collection.")
        del sessions[session_id]
        return {"message": "Session cleaned up."}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")
