#!/usr/bin/env python
# coding: utf-8

# # The real deal
# 

# In[15]:


import os
from datetime import datetime
from typing import Dict, List, Optional, TypedDict
import uuid
import tempfile

import assemblyai as aai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

import requests
from urllib.parse import urlparse
import mimetypes
import json


# In[16]:


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - Load API Keys from environment
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify keys are loaded
if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# Set up AssemblyAI - LAZY LOADING
# aai.settings.api_key = ASSEMBLYAI_API_KEY  # Moved to transcriber init


# Audio configuration
# SAMPLE_RATE = 16000
# CHANNELS = 1


# In[17]:


class TranscriptionState(TypedDict):
    """State for the transcription and summarization process"""
    session_id: str
    audio_input: Optional[str]  # original input (URL or path)
    audio_file_path: Optional[str]
    is_temp_file: bool  # flag for cleanup
    raw_transcript: str
    speaker_segments: List[Dict]
    final_transcript: str
    title: Optional[str]  # structured summary fields
    overview: Optional[str]
    key_points: Optional[str]
    action_items: Optional[str]
    important_details: Optional[str]
    error_message: Optional[str]
    processing_complete: bool

# In[27]:




#




# In[39]:


class AssemblyAITranscriber:
    """Transcriber using AssemblyAI SDK"""

    def __init__(self):
        # Initialize AssemblyAI only when transcriber is created (lazy loading)
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        self.transcriber = aai.Transcriber()

    def transcribe_with_speakers(self, audio_file: str) -> tuple:
        """Transcribe audio with speaker diarization using AssemblyAI SDK"""

        try:
            print("🎵 Starting transcription...")

            # Configure transcription with speaker labels
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=None,
                auto_chapters=False,
                sentiment_analysis=False,
                auto_highlights=False
            )

            # Transcribe the audio file
            print("⏳ Processing audio... This may take a moment...")
            transcript = self.transcriber.transcribe(audio_file, config)

            # Check if transcription was successful
            if transcript.status == aai.TranscriptStatus.error:
                print(f"❌ Transcription failed: {transcript.error}")
                return "", "", []

            # Get the full transcript text
            full_text = transcript.text
            print("✅ Transcription completed!")

            # Process speaker-labeled utterances
            speaker_segments = []
            formatted_transcript = ""

            if transcript.utterances:
                print(f"🎭 Processing {len(transcript.utterances)} speaker utterances...")

                for utterance in transcript.utterances:
                    segment = {
                        "speaker": f"Speaker_{utterance.speaker}",
                        "text": utterance.text,
                        "confidence": utterance.confidence,
                        "start": utterance.start / 1000,  # Convert ms to seconds
                        "end": utterance.end / 1000
                    }
                    speaker_segments.append(segment)
                    formatted_transcript += f"Speaker_{utterance.speaker}: {utterance.text}\n\n"

                    print(f"🗣️ Speaker_{utterance.speaker}: {utterance.text[:100]}...")
            else:
                # Fallback if no speaker labels detected
                print("⚠️ No speaker labels detected, treating as single speaker")
                speaker_segments = [{
                    "speaker": "Speaker_A",
                    "text": full_text,
                    "confidence": 0.8,
                    "start": 0,
                    "end": 60  # Approximate
                }]
                formatted_transcript = f"Speaker_A: {full_text}\n\n"

            unique_speakers = len(set(seg['speaker'] for seg in speaker_segments))
            print(f"🎭 Detected {unique_speakers} unique speaker(s)")
            print(f"📝 Generated {len(speaker_segments)} segments")

            return full_text, formatted_transcript, speaker_segments

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return "", "", []

# In[40]:


# def record_audio(state: TranscriptionState) -> TranscriptionState:
#     """Node: Record audio from microphone"""

#     if not AUDIO_RECORDING_AVAILABLE:
#         return {
#             **state,
#             "error_message": "Audio recording not available in deployment environment",
#             "processing_complete": True
#         }

#     print("🎤 AUDIO RECORDING")
#     print("="*50)

#     try:
#         recorder = AudioRecorder()
#     except RuntimeError as e:
#         return {
#             **state,
#             "error_message": str(e),
#             "processing_complete": True
#         }

#     # Start recording
#     if not recorder.start_recording():
#         return {
#             **state,
#             "error_message": "Failed to start audio recording",
#             "processing_complete": True
#         }

#     print("🎙️ Recording in progress...")
#     print("🔴 Press Enter when finished speaking")

#     # Wait for user to press Enter
#     try:
#         input()
#     except KeyboardInterrupt:
#         print("\n⚠️ Recording cancelled")
#         return {
#             **state,
#             "error_message": "Recording cancelled by user",
#             "processing_complete": True
#         }

#     # Stop recording and save file
#     audio_file = recorder.stop_recording()

#     if not audio_file:
#         return {
#             **state,
#             "error_message": "Failed to save audio recording",
#             "processing_complete": True
#         }

#     print("✅ Recording completed and saved")

#     return {
#         **state,
#         "audio_file_path": audio_file,
#         "error_message": None
#     }




# # In[41]:

# # Fix the transcribe_with_speakers function - DON'T clean up temp file here
# def transcribe_with_speakers(state: TranscriptionState) -> TranscriptionState:
#     """Node: Transcribe audio with speaker diarization using AssemblyAI SDK"""

#     if state.get("error_message"):
#         return state

#     audio_file = state.get("audio_file_path")
#     if not audio_file:
#         return {
#             **state,
#             "error_message": "No audio file available for transcription",
#             "processing_complete": True
#         }

#     print("\n🎵 TRANSCRIPTION WITH SPEAKER DIARIZATION")
#     print("="*50)

#     transcriber = AssemblyAITranscriber()

#     # Transcribe with speaker diarization
#     raw_transcript, formatted_transcript, speaker_segments = transcriber.transcribe_with_speakers(audio_file)

#     # DON'T clean up temp file here - it will be cleaned up in display_results_with_cleanup
#     # Only clean up if it's NOT a temp file (i.e., it was a local file used for recording)
#     if not state.get("is_temp_file"):
#         try:
#             os.unlink(audio_file)
#             print("✅ Local recording file cleaned up")
#         except:
#             pass

#     if not raw_transcript.strip():
#         return {
#             **state,
#             "error_message": "No speech detected in audio",
#             "processing_complete": True
#         }

#     print(f"\n✅ Transcription processing completed!")

#     return {
#         **state,
#         "raw_transcript": raw_transcript,
#         "final_transcript": formatted_transcript,
#         "speaker_segments": speaker_segments,
#         "error_message": None
#     }


# In[42]:


# Add this new node BEFORE your existing nodes
def download_audio_node(state: TranscriptionState) -> TranscriptionState:
    """Node: Download audio from URL or validate local file"""
    
    print("🌐 AUDIO INPUT PROCESSING")
    print("="*50)
    
    audio_input = state.get("audio_input")
    
    if not audio_input:
        return {
            **state,
            "error_message": "No audio input provided",
            "processing_complete": True
        }
    
    # Check if it's a URL
    if audio_input.startswith(('http://', 'https://')):
        print(f"📡 URL detected: {audio_input}")
        
        try:
            # Validate URL format
            parsed_url = urlparse(audio_input)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
            
            print("⬇️ Downloading audio from URL...")
            
            # Download with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(audio_input, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Determine file extension
            file_extension = None
            
            # Try URL path first
            url_path = parsed_url.path
            if url_path:
                _, ext = os.path.splitext(url_path)
                if ext and ext.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']:
                    file_extension = ext
            
            # Try content-type header
            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'audio' in content_type:
                    ext = mimetypes.guess_extension(content_type)
                    if ext:
                        file_extension = ext
            
            # Default fallback
            if not file_extension:
                file_extension = '.mp3'
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_filename = temp_file.name
            
            # Download in chunks
            total_size = 0
            chunk_size = 8192
            
            print("📥 Downloading...")
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    temp_file.write(chunk)
                    total_size += len(chunk)
            
            temp_file.close()
            
            # Verify download
            if total_size == 0:
                os.unlink(temp_filename)
                raise ValueError("Downloaded file is empty")
            
            print(f"✅ Download successful!")
            print(f"📁 Temporary file: {temp_filename}")
            print(f"📊 File size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            
            return {
                **state,
                "audio_file_path": temp_filename,
                "is_temp_file": True,
                "error_message": None
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return {
                **state,
                "error_message": f"Failed to download audio from URL: {str(e)}",
                "processing_complete": True
            }
        except Exception as e:
            print(f"❌ Download error: {e}")
            return {
                **state,
                "error_message": f"Error processing URL: {str(e)}",
                "processing_complete": True
            }
    
    else:
        # Local file path
        print(f"📁 Local file detected: {audio_input}")
        
        if not os.path.exists(audio_input):
            print("❌ File not found")
            return {
                **state,
                "error_message": f"Local audio file not found: {audio_input}",
                "processing_complete": True
            }
        
        # Check file size
        file_size = os.path.getsize(audio_input)
        print(f"✅ File exists - Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        return {
            **state,
            "audio_file_path": audio_input,
            "is_temp_file": False,
            "error_message": None
        }
    
    

def transcribe_with_speakers(state: TranscriptionState) -> TranscriptionState:
    """Node: Transcribe audio with speaker diarization using AssemblyAI SDK"""

    if state.get("error_message"):
        return state

    audio_file = state.get("audio_file_path")
    if not audio_file:
        return {
            **state,
            "error_message": "No audio file available for transcription",
            "processing_complete": True
        }

    print("\n🎵 TRANSCRIPTION WITH SPEAKER DIARIZATION")
    print("="*50)

    transcriber = AssemblyAITranscriber()

    # Transcribe with speaker diarization
    raw_transcript, formatted_transcript, speaker_segments = transcriber.transcribe_with_speakers(audio_file)

    if not raw_transcript.strip():
        return {
            **state,
            "error_message": "No speech detected in audio",
            "processing_complete": True
        }

    print(f"\n✅ Transcription processing completed!")

    return {
        **state,
        "raw_transcript": raw_transcript,
        "final_transcript": formatted_transcript,
        "speaker_segments": speaker_segments,
        "error_message": None
    }


def generate_structured_summary(state: TranscriptionState) -> TranscriptionState:
    """Node: Generate structured summary from transcript"""

    if state.get("error_message"):
        return state

    print("\n🎯 GENERATING STRUCTURED SUMMARY")
    print("="*50)

    try:
        client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )

        transcript = state.get("final_transcript", "")
        speaker_segments = state.get("speaker_segments", [])

        if not transcript.strip():
            return {
                **state,
                "title": "Empty Transcript",
                "overview": "No transcript available to summarize.",
                "key_points": "No content to analyze",
                "action_items": "No action items identified",
                "important_details": "No details available",
                "processing_complete": True
            }

        print("🤖 Generating title...")
        # 1. Generate title
        title_prompt = f"Generate a concise, descriptive title (maximum 6 words) for this conversation:\n\n{transcript[:300]}..."
        title_msg = HumanMessage(content=title_prompt)
        title_response = client.invoke([
            SystemMessage(content="Return only a short, descriptive title. No quotes, explanations, or extra text."), 
            title_msg
        ]).content.strip().strip('"').strip("'")
        
        print("📋 Generating overview...")
        # 2. Generate overview
        overview_prompt = f"Provide a brief 2-3 sentence overview of this conversation including speakers, type, and context:\n\n{transcript}"
        overview_msg = HumanMessage(content=overview_prompt)
        overview_response = client.invoke([
            SystemMessage(content="Return a concise 2-3 sentence overview of the conversation. Focus on who is speaking, what type of conversation it is, and the main context."), 
            overview_msg
        ]).content
        
        print("🎯 Extracting key points...")
        # 3. Extract key points
        keypoints_prompt = f"List the 3-5 most important points discussed in this conversation:\n\n{transcript}"
        keypoints_msg = HumanMessage(content=keypoints_prompt)
        keypoints_response = client.invoke([
            SystemMessage(content="Return a numbered list of key points (1. 2. 3. etc.), one point per line. Be specific and concise."), 
            keypoints_msg
        ]).content
        
        print("✅ Identifying action items...")
        # 4. Extract action items
        actions_prompt = f"Identify any action items, tasks, decisions, or next steps mentioned in this conversation:\n\n{transcript}"
        actions_msg = HumanMessage(content=actions_prompt)
        actions_response = client.invoke([
            SystemMessage(content="List any action items, tasks, or next steps using bullet points (•). If none are mentioned, return 'No specific action items identified.'"), 
            actions_msg
        ]).content
        
        print("📌 Extracting important details...")
        # 5. Extract important details
        details_prompt = f"Extract important names, dates, numbers, locations, or other key details mentioned in this conversation:\n\n{transcript}"
        details_msg = HumanMessage(content=details_prompt)
        details_response = client.invoke([
            SystemMessage(content="List important details like names, dates, amounts, locations, etc. using bullet points (•). If none are significant, return 'No specific details to highlight.'"), 
            details_msg
        ]).content

        print("✅ Structured summary generation completed!")

        return {
            **state,
            "title": title_response,
            "overview": overview_response,
            "key_points": keypoints_response,
            "action_items": actions_response,
            "important_details": details_response,
            "processing_complete": True,
            "error_message": None
        }

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return {
            **state,
            "error_message": f"Summary generation error: {str(e)}",
            "processing_complete": True
        }


# In[43]:


def display_results_with_cleanup(state: TranscriptionState) -> TranscriptionState:
    """Node: Display final results and cleanup temp files"""

    # print("\n" + "="*80)
    # print("🎉 TRANSCRIPTION & SUMMARY COMPLETE")
    # print("="*80)

    # if state.get("error_message"):
    #     print(f"❌ Error: {state['error_message']}")
    #     return state

    # # Display speaker information
    # speaker_segments = state.get("speaker_segments", [])
    # if speaker_segments:
    #     unique_speakers = set(seg["speaker"] for seg in speaker_segments)
    #     print(f"🎤 Speakers Detected: {len(unique_speakers)} ({', '.join(unique_speakers)})")
    #     print(f"📝 Total Segments: {len(speaker_segments)}")

    # # Display structured summary
    # print(f"\n📋 SUMMARY RESULTS:")
    # print("-" * 60)
    # print(f"Title: {state.get('title', 'N/A')}")
    # print(f"\nOverview:\n{state.get('overview', 'N/A')}")
    # print(f"\nKey Points:\n{state.get('key_points', 'N/A')}")
    # print(f"\nAction Items:\n{state.get('action_items', 'N/A')}")
    # print(f"\nImportant Details:\n{state.get('important_details', 'N/A')}")

    # Cleanup temp file if needed
    if state.get("is_temp_file") and state.get("audio_file_path"):
        try:
            os.unlink(state["audio_file_path"])
            print("\n🧹 Temporary file cleaned up successfully")
        except Exception as e:
            print(f"\n⚠️ Could not clean up temporary file: {e}")

    return {
        **state,
        "processing_complete": True
    }


# In[44]:




def create_enhanced_transcription_graph():
    """Create the enhanced LangGraph workflow with URL support and structured output"""

    workflow = StateGraph(TranscriptionState)

    # Add nodes
    workflow.add_node("download", download_audio_node)
    workflow.add_node("transcribe", transcribe_with_speakers)
    workflow.add_node("summarize", generate_structured_summary)
    workflow.add_node("display", display_results_with_cleanup)

    # Add edges
    workflow.add_edge(START, "download")
    workflow.add_edge("download", "transcribe")
    workflow.add_edge("transcribe", "summarize")
    workflow.add_edge("summarize", "display")
    workflow.add_edge("display", END)

    return workflow.compile()



# Replace your process_audio function with this enhanced version
def process_audio(audio_input):
    """Process audio using the enhanced LangGraph workflow with URL support"""
    
    if not audio_input or not isinstance(audio_input, str):
        return {
            "success": False,
            "dev_message": "Invalid audio input provided",
            "user_message": "Please provide a valid audio file path or URL",
            "payload": {},
        }
    
    print(f"🚀 Starting LangGraph audio processing...")
    print(f"📥 Input: {audio_input}")
    
    # Initialize enhanced state
    initial_state = {
        "session_id": str(uuid.uuid4()),
        "audio_input": audio_input.strip(),
        "audio_file_path": None,
        "is_temp_file": False,
        "raw_transcript": "",
        "speaker_segments": [],
        "final_transcript": "",
        "title": None,
        "overview": None,
        "key_points": None,
        "action_items": None,
        "important_details": None,
        "error_message": None,
        "processing_complete": False
    }
    
    # Create and run the enhanced graph
    graph = create_enhanced_transcription_graph()
    
    try:
        final_state = graph.invoke(initial_state)
        
        if final_state.get("error_message"):
            return {
                "success": False,
                "dev_message": final_state.get("error_message"),
                "user_message": "Something went wrong during processing. Please try again.",
                "payload": {},
            }
        
        # Return structured response
        speaker_count = len(set(seg["speaker"] for seg in final_state.get("speaker_segments", [])))
        
        return {
            "success": True,
            "dev_message": "LangGraph processing completed successfully",
            "user_message": "Your audio has been processed and summarized successfully!",
            "payload": {
                "title": final_state.get("title", "Audio Summary"),
                "overview": final_state.get("overview", "No overview available"),
                "key_points": final_state.get("key_points", "No key points identified"),
                "action_items": final_state.get("action_items", "No action items identified"),
                "important_details": final_state.get("important_details", "No important details identified"),
                "transcript": final_state.get("final_transcript", "No transcript available"),
                "metadata": {
                    "session_id": final_state.get("session_id"),
                    "speaker_count": speaker_count,
                    "transcript_length": len(final_state.get("final_transcript", "")),
                    "processing_timestamp": datetime.now().isoformat(),
                    "input_type": "url" if audio_input.startswith(('http', 'https')) else "local_file",
                    "original_input": audio_input
                }
            },
        }
        
    except Exception as e:
        print(f"❌ LangGraph processing error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "dev_message": f"LangGraph processing error: {str(e)}",
            "user_message": "An unexpected error occurred during processing. Please try again.",
            "payload": {},
        }



# In[46]:

# if __name__ == "__main__":
#     # Test function - not used in API
#     test_url = "https://example.com/test.mp3"
#     result = process_audio(test_url)
#     print(result)


# if __name__ == "__main__":
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#     main()


# In[1]:


# In[ ]:

#A funtion to take in audio file and then return summary.
#I need to add this to the end of the Real_Deal.py file


# Remove this problematic code:
# if __name__ == "__main__":
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY