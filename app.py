import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Educational Image Analyzer",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f9f9f9;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_summary' not in st.session_state:
        st.session_state.processed_summary = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def setup_groq_llm():
    """Setup Groq LLM with API key"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        return None
    
    try:
        llm = ChatGroq(model="gemma2-9b-it")
        return llm
    except Exception as e:
        st.error(f"❌ Error initializing Groq LLM: {str(e)}")
        return None

def hindi_summarization(image_files):
    """Process uploaded images and generate educational summary"""
    
    # Setup LLM
    llm = setup_groq_llm()
    if not llm:
        return None
    
    try:
        # Load documents from images
        docs = []
        for image_file in image_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp_file:
                tmp_file.write(image_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                loader = UnstructuredImageLoader(tmp_file_path)
                docs.extend(loader.load())
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not docs:
            st.error("❌ No text could be extracted from the uploaded images.")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        
        # Create prompts
        chunk_prompt = """ 
        Please summarize the below story 
        Story: {text}
        """
        
        chunk_prompt_template = PromptTemplate(input_variables=['text'], template=chunk_prompt)
        
        final_prompt = """ 
        You are a teacher who is teaching a class of students.
        You are given a story and you need to explain it to the students in hindi.
        story: {text}
        """
        
        final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)
        
        # Create and run summary chain
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=chunk_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=False
        )
        
        # Run the chain
        with st.spinner("🤖 AI is analyzing your images and generating educational content..."):
            output = summary_chain.invoke({"input_documents": texts})
        
        return output.get('output_text', '')
        
    except Exception as e:
        st.error(f"❌ Error processing images: {str(e)}")
        return None


def process_images(image_files):
    """Process uploaded images and generate educational summary"""
    
    # Setup LLM
    llm = setup_groq_llm()
    if not llm:
        return None
    
    try:
        # Load documents from images
        docs = []
        for image_file in image_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp_file:
                tmp_file.write(image_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                loader = UnstructuredImageLoader(tmp_file_path)
                docs.extend(loader.load())
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not docs:
            st.error("❌ No text could be extracted from the uploaded images.")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        
        # Create prompts
        chunk_prompt = """ 
        Please summarize the below story 
        Story: {text}
        """
        
        chunk_prompt_template = PromptTemplate(input_variables=['text'], template=chunk_prompt)
        
        final_prompt = """ 
        You are a teacher who is teaching a class of students.
        You are given a story and you need to explain it to the students.
        story: {text}
        """
        
        final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)
        
        # Create and run summary chain
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=chunk_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=False
        )
        
        # Run the chain
        with st.spinner("🤖 AI is analyzing your images and generating educational content..."):
            output = summary_chain.invoke({"input_documents": texts})
        
        return output.get('output_text', '')
        
    except Exception as e:
        st.error(f"❌ Error processing images: {str(e)}")
        return None

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Educational Image Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload images containing text to generate educational summaries and analysis</p>', unsafe_allow_html=True)
        
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Images")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose images with text content",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload images containing text that you want to analyze"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Display uploaded files
            st.subheader("📁 Uploaded Files")
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. {file.name} ({file.size} bytes)")
            
            # Process button
            if st.button("🚀 Process Images", type="primary", use_container_width=True):
                if not os.getenv("GROQ_API_KEY"):
                    st.error("❌ Please set up your GROQ API key in the .env file first.")
                else:
                    # Process images
                    summary = process_images(uploaded_files)
                    hindi_summary = hindi_summarization(uploaded_files)
                    
                    if summary:
                        st.session_state.processed_summary = summary
                        st.session_state.hindi_summary = hindi_summary
                        st.session_state.uploaded_files = uploaded_files
                        st.session_state.processing_complete = True
                        st.success("✅ Processing completed successfully!")
                        st.rerun()
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.processing_complete and st.session_state.processed_summary:
            # Display the summary
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.subheader("🎓 Educational Summary")
            st.write(st.session_state.processed_summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")

            # Display the Hindi summary
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.subheader("🎓 Educational Summary in Hindi")
            st.write(st.session_state.hindi_summary)
            st.markdown('</div>', unsafe_allow_html=True)

            # Additional options
            st.markdown("---")
            col_download, col_clear = st.columns(2)
            
            with col_download:
                # Download summary as text file
                final_summary = st.session_state.processed_summary + st.session_state.hindi_summary
                st.download_button(
                    label="💾 Download Summary",
                    data=final_summary,
                    file_name=f"educational_summary_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_clear:
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state.processed_summary = None
                    st.session_state.uploaded_files = []
                    st.session_state.processing_complete = False
                    st.rerun()
        
        elif st.session_state.uploaded_files and not st.session_state.processing_complete:
            st.info("👆 Click 'Process Images' to generate the educational summary")
        
        else:
            st.info("📤 Upload some images to get started!")

if __name__ == "__main__":
    main()
