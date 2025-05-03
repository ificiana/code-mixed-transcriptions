import streamlit as st

def main():
    st.title("Audio Processing App")
    
    st.markdown("""
    Welcome to the Audio Processing App! This application provides tools for working with audio files:
    
    ### 🎵 Features
    
    1. **Audio Chunking**
       - Split long audio files into smaller chunks
       - Customize chunk duration
       - Download individual chunks
    
    2. **Voice Isolation**
       - Isolate vocals from audio files
       - Remove background noise
       - Multiple model options for different quality/speed tradeoffs
    
    ### 📝 Getting Started
    
    Choose a tool to get started:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 Audio Chunking", use_container_width=True, help="Split long recordings into smaller segments"):
            st.switch_page("1_audio_chunking.py")
    
    with col2:
        if st.button("🎤 Voice Isolation", use_container_width=True, help="Extract clean vocals from mixed audio"):
            st.switch_page("2_voice_isolation.py")
            
    st.markdown("""
    """)
    
    st.markdown("---")
    st.markdown("Built with Streamlit and Demucs")

if __name__ == "__main__":
    main()
