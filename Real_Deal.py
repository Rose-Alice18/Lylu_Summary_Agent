#!/usr/bin/env python
# coding: utf-8

# # The real deal
# 

# In[15]:


import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, TypedDict
import uuid
import numpy as np
import tempfile
import wave

import sounddevice as sd
import assemblyai as aai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START


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


# Set up AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1


# In[17]:


class TranscriptionState(TypedDict):
    """State for the transcription and summarization process"""
    session_id: str
    audio_file_path: Optional[str]
    raw_transcript: str
    speaker_segments: List[Dict]
    final_transcript: str
    summary_report: Optional[str]
    error_message: Optional[str]
    processing_complete: bool


# In[27]:


class AudioRecorder:
    """Reliable audio recorder for AssemblyAI"""

    def __init__(self):
        self.is_recording = False
        self.audio_data = []

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_data = []

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            if self.is_recording:
                self.audio_data.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            print("🎤 Recording started...")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording and create WAV file"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if not self.audio_data:
            print(" No audio data recorded")
            return None

        # Combine all audio chunks
        full_audio = np.concatenate(self.audio_data, axis=0).flatten()

        # Convert to int16 (WAV standard)
        audio_int16 = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)

        # Create WAV file using wave module (most compatible)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()

        try:
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())

            # Verify the file
            file_size = os.path.getsize(temp_filename)
            duration = len(audio_int16) / SAMPLE_RATE

            print(f" Audio saved: {temp_filename}")
            print(f" File size: {file_size} bytes")
            print(f" Duration: {duration:.1f} seconds")

            return temp_filename

        except Exception as e:
            print(f"❌ Failed to create WAV file: {e}")
            return None




# In[39]:


class AssemblyAITranscriber:
    """Transcriber using AssemblyAI SDK"""

    def __init__(self):
        self.transcriber = aai.Transcriber()

    def transcribe_with_speakers(self, audio_file: str) -> tuple:
        """Transcribe audio with speaker diarization using AssemblyAI SDK"""

        try:
            print("🎵 Starting transcription...")

            # Configure transcription with speaker labels
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=None,  # Can be adjusted44
                auto_chapters=False,
                sentiment_analysis=False,
                auto_highlights=False
            )

            # Transcribe the audio file
            print(" Processing audio... This may take a moment...")
            transcript = self.transcriber.transcribe(audio_file, config)

            # Check if transcription was successful
            if transcript.status == aai.TranscriptStatus.error:
                print(f" Transcription failed: {transcript.error}")
                return "", "", []

            # Get the full transcript text
            full_text = transcript.text
            print(" Transcription completed!")

            # Process speaker-labeled utterances
            speaker_segments = []
            formatted_transcript = ""

            if transcript.utterances:
                print(f" Processing {len(transcript.utterances)} speaker utterances...")

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

                    print(f" Speaker_{utterance.speaker}: {utterance.text[:100]}...")
            else:
                # Fallback if no speaker labels detected
                print(" No speaker labels detected, treating as single speaker")
                speaker_segments = [{
                    "speaker": "Speaker_A",
                    "text": full_text,
                    "confidence": 0.8,
                    "start": 0,
                    "end": 60  # Approximate
                }]
                formatted_transcript = f"Speaker_A: {full_text}\n\n"

            unique_speakers = len(set(seg['speaker'] for seg in speaker_segments))
            print(f" Detected {unique_speakers} unique speaker(s)")
            print(f" Generated {len(speaker_segments)} segments")

            return full_text, formatted_transcript, speaker_segments

        except Exception as e:
            print(f" Transcription error: {e}")
            return "", "", []


# In[40]:


def record_audio(state: TranscriptionState) -> TranscriptionState:
    """Node: Record audio from microphone"""

    print("🎤 AUDIO RECORDING")
    print("="*50)

    recorder = AudioRecorder()

    # Start recording
    if not recorder.start_recording():
        return {
            **state,
            "error_message": "Failed to start audio recording",
            "processing_complete": True
        }

    print("🎙️ Recording in progress...")
    print("🔴 Press Enter when finished speaking")

    # Wait for user to press Enter
    try:
        input()
    except KeyboardInterrupt:
        print("\n Recording cancelled")
        return {
            **state,
            "error_message": "Recording cancelled by user",
            "processing_complete": True
        }

    # Stop recording and save file
    audio_file = recorder.stop_recording()

    if not audio_file:
        return {
            **state,
            "error_message": "Failed to save audio recording",
            "processing_complete": True
        }

    print("✅ Recording completed and saved")

    return {
        **state,
        "audio_file_path": audio_file,
        "error_message": None
    }


# In[41]:


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

    print("\n TRANSCRIPTION WITH SPEAKER DIARIZATION")
    print("="*50)

    transcriber = AssemblyAITranscriber()

    # Transcribe with speaker diarization
    raw_transcript, formatted_transcript, speaker_segments = transcriber.transcribe_with_speakers(audio_file)

    # Clean up temp file
    try:
        os.unlink(audio_file)
        print(" Temporary audio file cleaned up")
    except:
        pass

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


# In[42]:


def generate_summary(state: TranscriptionState) -> TranscriptionState:
    """Node: Generate structured summary from transcript"""

    if state.get("error_message"):
        return state

    print("\n GENERATING SUMMARY")
    print("="*50)

    try:
        client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )

        transcript = state.get("final_transcript", "")
        speaker_segments = state.get("speaker_segments", [])

        if not transcript.strip():
            return {
                **state,
                "summary_report": "  CONVERSATION SUMMARY\n\n No transcript available to summarize.",
                "processing_complete": True
            }

        system_message = SystemMessage(content="""
You are an expert conversation analyzer and an intelligent conversation summarizer. Create a comprehensive, structured summary of the provided conversation transcript that includes speaker diarization.

**SPECIAL INSTRUCTION FOR MEDICAL CONVERSATIONS:**
If the conversation appears to be a medical visit, consultation, or healthcare-related discussion, additionally provide a structured MEDICAL VISIT SUMMARY section using this format or you can improvise too:

 MEDICAL VISIT SUMMARY

 PATIENT INFORMATION
- Patient Name: [Extract from conversation]
- Date of Birth: [If mentioned]
- Medical Record Number: [If mentioned]

 VISIT DETAILS
- Date of Visit: [Extract or use transcript date]
- Provider: [Doctor/healthcare provider name]
- Location: [Clinic/hospital name if mentioned]
- Visit Type: [Routine, follow-up, urgent, etc.]
- Duration: [If determinable from conversation]

 REASON FOR VISIT
- Primary: [Main reason for the visit]
- Secondary: [Additional concerns or follow-ups]
- Patient concerns: [Any symptoms or concerns mentioned]

 DIAGNOSES & CONDITIONS
[List all medical conditions discussed with current status]
1. [Condition name] - [Status: controlled/uncontrolled/new/resolved]
2. [Additional conditions as discussed]

 MEDICATIONS DISCUSSED
- Current medications: [List medications mentioned]
- New prescriptions: [Any new medications prescribed]
- Medication changes: [Any adjustments discussed]

 TESTS & PROCEDURES
- Tests ordered: [Any lab work, imaging, etc.]
- Results reviewed: [Any test results discussed]
- Vital signs: [If mentioned in conversation]

 RECOMMENDATIONS & PLAN
- Treatment plan: [Specific recommendations given]
- Lifestyle modifications: [Diet, exercise, etc.]
- Follow-up instructions: [When to return, what to monitor]

 FOLLOW-UP
- Next appointment: [Date/timeframe if mentioned]
- When to call: [Circumstances requiring contact]
- Monitoring instructions: [Home monitoring, etc.]

 IMPORTANT NOTES
- Allergies: [If discussed]
- Emergency instructions: [If provided]
- Patient questions: [Questions to address at next visit]


You can also add this on to the report with the following sections:

 📋 CONVERSATION OVERVIEW
- Duration and context
- 
- Number of speakers identified. Now, identify the speakers and their roles if possible (e.g., interviewer/interviewee, doctor/patient, etc.)
- Type of conversation (meeting, interview, discussion, etc.)

 🎯 KEY POINTS SUMMARY
- Main topics discussed
- Important decisions made
- Critical information shared

 👥 INDIVIDUAL SPEAKER CONTRIBUTIONS
For each speaker, provide:
- Their main contributions
- Key points they raised
- Their role/perspective in the conversation

 💡 NOTABLE IDEAS & INSIGHTS
- Creative or innovative ideas mentioned
- Important insights or revelations
- Unique perspectives shared

 ✅ ACTION ITEMS & NEXT STEPS
- Specific actions mentioned
- Deadlines or timelines discussed
- Follow-up items identified

 📌 IMPORTANT DETAILS
- Names, dates, numbers mentioned
- Resources or references cited
- Contact information or links

Format the summary professionally with clear headers and bullet points. Be concise but comprehensive.

Requirements:
- Use clear, professional language appropriate to the content
- Maintain confidentiality (use generic terms instead of personal names when appropriate)
- Include relevant information, main topics, key decisions, and action items
- If the content is medical, use medical terminology; if it's business, use business language, etc.
- Highlight any urgent or important information
- If the content is unclear or contains no meaningful information, note this in the summary
- Adapt the summary style to match the content type (medical, business, personal, educational, etc.)

Format the summary with clear sections and bullet points where appropriate.
""")

        # Prepare detailed speaker analysis
        speaker_info = ""
        if speaker_segments:
            unique_speakers = set(seg["speaker"] for seg in speaker_segments)
            speaker_info = f"\n\nSPEAKER ANALYSIS:\n"
            speaker_info += f"Total unique speakers detected: {len(unique_speakers)}\n"
            speaker_info += f"Total segments: {len(speaker_segments)}\n\n"

            for speaker in unique_speakers:
                speaker_segs = [seg for seg in speaker_segments if seg["speaker"] == speaker]
                total_words = sum(len(seg["text"].split()) for seg in speaker_segs)
                total_duration = sum(seg.get("end", 0) - seg.get("start", 0) for seg in speaker_segs)
                avg_confidence = sum(seg.get("confidence", 0) for seg in speaker_segs) / len(speaker_segs) if speaker_segs else 0

                speaker_info += f"{speaker}:\n"
                speaker_info += f"  - {len(speaker_segs)} segments\n"
                speaker_info += f"  - ~{total_words} words\n"
                speaker_info += f"  - {total_duration:.1f}s total speaking time\n"
                speaker_info += f"  - {avg_confidence:.2f} avg confidence\n\n"

        user_message = HumanMessage(content=f"""
Please analyze and summarize this conversation transcript with speaker diarization:

FULL TRANSCRIPT:
{transcript}
{speaker_info}

Create a structured report following the format specified in your instructions.
""")

        response = client.invoke([system_message, user_message])
        summary_report = response.content

        print("✅ Summary generated successfully!")

        return {
            **state,
            "summary_report": summary_report,
            "processing_complete": True,
            "error_message": None
        }

    except Exception as e:
        print(f" Error generating summary: {e}")
        return {
            **state,
            "error_message": f"Summary generation error: {str(e)}",
            "processing_complete": True
        }


# In[43]:


def display_results(state: TranscriptionState) -> TranscriptionState:
    """Node: Display final results"""

    print("\n" + "="*80)
    print(" TRANSCRIPTION & SUMMARY COMPLETE")
    print("="*80)

    if state.get("error_message"):
        print(f" Error: {state['error_message']}")
        return state

    # Display speaker information
    speaker_segments = state.get("speaker_segments", [])
    if speaker_segments:
        unique_speakers = set(seg["speaker"] for seg in speaker_segments)
        print(f" Speakers Detected: {len(unique_speakers)} ({', '.join(unique_speakers)})")
        print(f" Total Segments: {len(speaker_segments)}")

        # Show detailed speaker breakdown
        for speaker in unique_speakers:
            speaker_segs = [seg for seg in speaker_segments if seg["speaker"] == speaker]
            total_words = sum(len(seg["text"].split()) for seg in speaker_segs)
            total_time = sum(seg.get("end", 0) - seg.get("start", 0) for seg in speaker_segs)
            print(f"   {speaker}: {total_words} words, {total_time:.1f}s")

    # Display transcript
    print(f"\n FULL TRANSCRIPT:")
    print("-" * 60)
    transcript = state.get("final_transcript", "No transcript available")
    print(transcript)

    # Display summary
    print(f"\n GENERATED SUMMARY REPORT:")
    print("-" * 60)
    summary = state.get("summary_report", "No summary available")
    print(summary)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = state.get("session_id", "unknown")

    filename = f"assemblyai_transcript_summary_{timestamp}_{session_id[:8]}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ASSEMBLYAI TRANSCRIPTION & SUMMARY REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

            if speaker_segments:
                unique_speakers = set(seg["speaker"] for seg in speaker_segments)
                f.write(f"Speakers: {len(unique_speakers)} ({', '.join(unique_speakers)})\n")

                # Detailed speaker stats
                f.write("\nSPEAKER STATISTICS:\n")
                for speaker in unique_speakers:
                    speaker_segs = [seg for seg in speaker_segments if seg["speaker"] == speaker]
                    total_words = sum(len(seg["text"].split()) for seg in speaker_segs)
                    total_time = sum(seg.get("end", 0) - seg.get("start", 0) for seg in speaker_segs)
                    avg_conf = sum(seg.get("confidence", 0) for seg in speaker_segs) / len(speaker_segs)
                    f.write(f"{speaker}: {total_words} words, {total_time:.1f}s, {avg_conf:.2f} confidence\n")
            else:
                f.write("Speakers: None detected\n")

            f.write("\n" + "="*60 + "\n")
            f.write("TRANSCRIPT:\n")
            f.write("="*60 + "\n")
            f.write(transcript + "\n")
            f.write("\n" + "="*60 + "\n")
            f.write("SUMMARY REPORT:\n")
            f.write("="*60 + "\n")
            f.write(summary + "\n")

        print(f"\n Results saved to: {filename}")

    except Exception as e:
        print(f" Could not save to file: {e}")

    return {
        **state,
        "processing_complete": True
    }


# In[44]:


def create_transcription_graph():
    """Create the LangGraph workflow for transcription and summarization"""

    workflow = StateGraph(TranscriptionState)

    # Add nodes
    workflow.add_node("record", record_audio)
    workflow.add_node("transcribe", transcribe_with_speakers)
    workflow.add_node("summarize", generate_summary)
    workflow.add_node("display", display_results)

    # Add edges
    workflow.add_edge(START, "record")
    workflow.add_edge("record", "transcribe")
    workflow.add_edge("transcribe", "summarize")
    workflow.add_edge("summarize", "display")
    workflow.add_edge("display", END)

    return workflow.compile()


# In[45]:


def main():
    """Main execution function"""

    print(" ASSEMBLYAI SDK TRANSCRIPTION AGENT")
    print("="*70)
    print("Features:")
    print("• Record audio from microphone")
    print("• Professional speaker diarization")
    print("• High-accuracy transcription")
    print("• Detailed speaker analysis")
    print("• Automatic summary generation")
    print("• Structured report output")
    print("="*70)

    # Verify API keys
    if not ASSEMBLYAI_API_KEY or len(ASSEMBLYAI_API_KEY) < 20:
        print(" AssemblyAI API key appears to be invalid")
        print(" Get your API key from: https://www.assemblyai.com/dashboard/")
        return

    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        print(" OpenAI API key appears to be invalid")
        return

    print(" API keys configured")

    # Test audio devices
    print("\n Available audio devices:")
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            for device in input_devices[:3]:
                print(f"  ✓ {device['name']}")
        else:
            print("   No input devices found!")
            return
    except Exception as e:
        print(f" Could not query audio devices: {e}")
        return

    # Initialize state
    initial_state = {
        "session_id": str(uuid.uuid4()),
        "audio_file_path": None,
        "raw_transcript": "",
        "speaker_segments": [],
        "final_transcript": "",
        "summary_report": None,
        "error_message": None,
        "processing_complete": False
    }

    # Create and run the graph
    graph = create_transcription_graph()

    try:
        final_state = graph.invoke(initial_state)

        print("\n" + "="*60)
        if final_state.get("error_message"):
            print(" PROCESSING COMPLETED WITH ERRORS")
            print(f"Error: {final_state.get('error_message')}")
        else:
            print(" PROCESSING COMPLETED SUCCESSFULLY!")

        print("="*60)
        print(f"Session ID: {final_state.get('session_id')}")
        print(f"Transcript Length: {len(final_state.get('final_transcript', ''))}")
        print(f"Summary Generated: {'Yes' if final_state.get('summary_report') else 'No'}")

        speaker_segments = final_state.get('speaker_segments', [])
        unique_speakers = set(seg['speaker'] for seg in speaker_segments)
        print(f"Speakers Detected: {len(unique_speakers)}")

    except KeyboardInterrupt:
        print("\n Process interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# In[46]:


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    main()


# In[1]:





# In[ ]:




