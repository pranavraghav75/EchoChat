import streamlit as st
import requests
from typing import Dict, Optional

def reset_session_state():
    st.session_state.session_id = None
    st.session_state.celebrity = ""
    st.session_state.general_info = ""
    st.session_state.chat_history = []
    st.session_state.message_key = 0
    st.session_state.personality = ""

def main():
    st.set_page_config(page_title="EchoAI", layout="wide")
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'general_info' not in st.session_state:
        st.session_state.general_info = ""
    if 'message_key' not in st.session_state:
        st.session_state.message_key = 0
    if 'personality' not in st.session_state:
        st.session_state.personality = ""

    st.title("EchoChat")

    if not st.session_state.session_id:
        celebrity = st.text_input("Enter Celebrity Name", key="celebrity_input")

        if st.button("Start Chat"):
            if celebrity:
                with st.spinner("Initializing chat..."):
                    try:
                        response = requests.post(
                            "http://localhost:8000/init_session",
                            params={"celebrity": celebrity}
                        )
                        if not response.ok:
                            st.error(f"Failed to initialize chat: {response.json().get('detail', 'Unknown error')}")
                            return
                        
                        data = response.json()
                        
                        st.session_state.session_id = data["session_id"]
                        st.session_state.celebrity = celebrity
                        st.session_state.general_info = data.get("data", "")
                        st.session_state.personality = data.get("personality", "")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error initializing session: {str(e)}")

        if st.session_state.general_info:
            st.subheader("General Info:")
            st.write(st.session_state.general_info)

    else:
        st.title(f"Chat with {st.session_state.celebrity}")

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                st.markdown(f"**You:** {msg['user']}")
                st.markdown(f"**{st.session_state.celebrity}:** {msg['bot']}")

        col1, col2 = st.columns([4, 1])
        with col1:
            user_message = st.text_input("Your message", key=f"message_input_{st.session_state.message_key}")
        
        with col2:
            if st.button("Send"):
                if user_message:
                    with st.spinner("Sending message..."):
                        try:
                            response = requests.post(
                                "http://localhost:8000/chat",
                                params={
                                    "session_id": st.session_state.session_id,
                                    "user_message": user_message
                                }
                            )
                            
                            if response.status_code == 404:
                                st.error("Session expired. Please start a new chat.")
                                reset_session_state()
                                st.rerun()
                                return
                            
                            if not response.ok:
                                st.error(f"Server error: {response.json().get('detail', 'Unknown error')}")
                                return
                            
                            response_data = response.json()
                            
                            if 'answer' not in response_data:
                                st.error("Invalid response format from server")
                                return
                            
                            bot_response = response_data['answer']
                            
                            st.session_state.chat_history.append({
                                "user": user_message,
                                "bot": bot_response
                            })
                            
                            st.session_state.message_key += 1
                            
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error: {str(e)}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        if st.button("End Session"):
            try:
                response = requests.post(
                    "http://localhost:8000/cleanup_session",
                    params={"session_id": st.session_state.session_id}
                )
                reset_session_state()
                st.rerun()
            except Exception as e:
                st.error(f"Error cleaning up session: {str(e)}")
                reset_session_state()
                st.rerun()

        if st.session_state.personality:
            with st.sidebar:
                st.subheader("💫 Personality Insights")
                st.markdown(st.session_state.personality)

if __name__ == "__main__":
    main()