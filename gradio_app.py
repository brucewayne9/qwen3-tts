"""Gradio interface for Qwen3-TTS voice management."""
import os
import requests
import gradio as gr
from pathlib import Path

RESOURCES_DIR = Path("/home/aialfred/qwen3-tts/resources")
# Use environment variable or default to localhost
API_URL = os.environ.get("QWEN_TTS_API_URL", "http://localhost:7860")


def list_voices():
    """List all available cloned voices."""
    voices = []
    for f in RESOURCES_DIR.iterdir():
        if f.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']:
            voices.append(f.stem)
    return list(set(voices))  # Remove duplicates


def list_designed_voices():
    """List all designed voices from API."""
    try:
        resp = requests.get(f"{API_URL}/voice_design/voices", timeout=10)
        if resp.status_code == 200:
            return [v["name"] for v in resp.json().get("voices", [])]
    except:
        pass
    return []


def get_designed_voice_info(name):
    """Get info about a designed voice."""
    try:
        resp = requests.get(f"{API_URL}/voice_design/voices", timeout=10)
        if resp.status_code == 200:
            for v in resp.json().get("voices", []):
                if v["name"] == name:
                    return f"Description: {v['description']}\nLanguage: {v.get('language', 'English')}"
    except:
        pass
    return "No info available"


def upload_voice(audio_file, voice_name):
    """Upload a new voice sample."""
    if not audio_file:
        return "Please provide an audio file", list_voices()
    if not voice_name or not voice_name.strip():
        return "Please provide a voice name", list_voices()

    voice_name = voice_name.strip().replace(" ", "_")

    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (f'{voice_name}.wav', f, 'audio/wav')}
            data = {'audio_file_label': voice_name}
            resp = requests.post(f"{API_URL}/upload_audio/", files=files, data=data, timeout=30)

        if resp.status_code == 200:
            return f"Voice '{voice_name}' uploaded successfully!", list_voices()
        else:
            return f"Upload failed: {resp.text}", list_voices()
    except Exception as e:
        return f"Error: {e}", list_voices()


def test_voice(voice_name, text):
    """Test a voice with sample text."""
    if not voice_name:
        return None, "Please select a voice"
    if not text:
        text = "Hello, this is a test of the voice cloning system."

    try:
        resp = requests.get(
            f"{API_URL}/synthesize_speech/",
            params={"text": text, "voice": voice_name},
            timeout=60
        )
        if resp.status_code == 200:
            output_path = f"/tmp/test_voice_{voice_name}.wav"
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return output_path, f"Generated audio for voice: {voice_name}"
        else:
            return None, f"Error: {resp.text}"
    except Exception as e:
        return None, f"Error: {e}"


def delete_voice(voice_name):
    """Delete a voice from resources."""
    if not voice_name:
        return "Please select a voice", list_voices()

    deleted = []
    for f in RESOURCES_DIR.iterdir():
        if f.stem == voice_name:
            f.unlink()
            deleted.append(f.name)

    if deleted:
        return f"Deleted: {', '.join(deleted)}", list_voices()
    return f"Voice '{voice_name}' not found", list_voices()


# Build Gradio interface
with gr.Blocks(title="Qwen3-TTS Voice Manager") as app:
    gr.Markdown("# Qwen3-TTS Voice Manager")
    gr.Markdown("Upload voice samples for cloning, test voices, and manage your voice library.")

    with gr.Tab("Upload Voice"):
        gr.Markdown("### Upload a new voice sample (10-30 seconds of clear speech)")
        with gr.Row():
            audio_input = gr.Audio(label="Voice Sample", type="filepath")
            voice_name_input = gr.Textbox(label="Voice Name", placeholder="e.g., bruce, jarvis, sarah")
        upload_btn = gr.Button("Upload Voice", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Test Voice"):
        gr.Markdown("### Test a voice with sample text")
        with gr.Row():
            voice_dropdown = gr.Dropdown(choices=list_voices(), label="Select Voice", interactive=True)
            refresh_btn = gr.Button("🔄 Refresh")
        test_text = gr.Textbox(label="Text to speak", value="Hello, this is a test of the voice cloning system.")
        test_btn = gr.Button("Generate Speech", variant="primary")
        audio_output = gr.Audio(label="Generated Audio")
        test_status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Manage Voices"):
        gr.Markdown("### Current cloned voices in library")
        voice_list = gr.Dropdown(choices=list_voices(), label="Voices", interactive=True)
        with gr.Row():
            refresh_list_btn = gr.Button("🔄 Refresh List")
            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
        manage_status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Voice Design"):
        gr.Markdown("### Create voices from natural language descriptions")
        gr.Markdown("Describe a voice and the AI will generate it. No audio sample needed!")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Preview a Voice")
                preview_desc = gr.Textbox(
                    label="Voice Description",
                    placeholder="e.g., Male, 30 years old, deep voice, calm and professional tone",
                    lines=2
                )
                preview_lang = gr.Dropdown(
                    choices=["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian"],
                    value="English",
                    label="Language"
                )
                preview_text = gr.Textbox(
                    label="Preview Text",
                    value="Hello, this is a preview of the designed voice.",
                    lines=2
                )
                preview_btn = gr.Button("🎧 Preview Voice", variant="primary")
                preview_audio = gr.Audio(label="Preview Audio")
                preview_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                gr.Markdown("#### Save as Named Voice")
                save_name = gr.Textbox(label="Voice Name", placeholder="e.g., professional_narrator")
                save_desc = gr.Textbox(
                    label="Voice Description",
                    placeholder="e.g., Female, young, cheerful and energetic",
                    lines=2
                )
                save_lang = gr.Dropdown(
                    choices=["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian"],
                    value="English",
                    label="Language"
                )
                save_btn = gr.Button("💾 Save Voice", variant="primary")
                save_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("#### Saved Designed Voices")
        with gr.Row():
            designed_voice_list = gr.Dropdown(choices=list_designed_voices(), label="Designed Voices", interactive=True)
            refresh_designed_btn = gr.Button("🔄 Refresh")
        designed_voice_info = gr.Textbox(label="Voice Info", interactive=False, lines=3)
        with gr.Row():
            test_designed_text = gr.Textbox(label="Test Text", value="Hello, this is a test.", scale=3)
            test_designed_btn = gr.Button("🎧 Test", variant="secondary")
        test_designed_audio = gr.Audio(label="Generated Audio")
        delete_designed_btn = gr.Button("🗑️ Delete Selected Voice", variant="stop")

    # Event handlers
    upload_btn.click(
        upload_voice,
        inputs=[audio_input, voice_name_input],
        outputs=[upload_status, voice_dropdown]
    )

    refresh_btn.click(
        lambda: gr.update(choices=list_voices()),
        outputs=[voice_dropdown]
    )

    test_btn.click(
        test_voice,
        inputs=[voice_dropdown, test_text],
        outputs=[audio_output, test_status]
    )

    refresh_list_btn.click(
        lambda: gr.update(choices=list_voices()),
        outputs=[voice_list]
    )

    delete_btn.click(
        delete_voice,
        inputs=[voice_list],
        outputs=[manage_status, voice_list]
    )

    # Voice Design event handlers
    def preview_voice_design(description, language, text):
        if not description:
            return None, "Please enter a voice description"
        try:
            resp = requests.get(
                f"{API_URL}/voice_design/preview",
                params={"description": description, "language": language, "text": text},
                timeout=120
            )
            if resp.status_code == 200:
                output_path = "/tmp/voice_design_preview.wav"
                with open(output_path, 'wb') as f:
                    f.write(resp.content)
                return output_path, "Preview generated successfully"
            else:
                return None, f"Error: {resp.text}"
        except Exception as e:
            return None, f"Error: {e}"

    def save_designed_voice(name, description, language):
        if not name or not description:
            return "Please provide both name and description"
        try:
            resp = requests.post(
                f"{API_URL}/voice_design/create",
                data={"name": name, "description": description, "language": language},
                timeout=30
            )
            if resp.status_code == 200:
                return f"Voice '{name}' saved successfully!"
            else:
                return f"Error: {resp.text}"
        except Exception as e:
            return f"Error: {e}"

    def test_designed_voice(voice_name, text):
        if not voice_name:
            return None
        try:
            resp = requests.get(
                f"{API_URL}/voice_design/synthesize",
                params={"voice": voice_name, "text": text},
                timeout=120
            )
            if resp.status_code == 200:
                output_path = f"/tmp/designed_voice_test.wav"
                with open(output_path, 'wb') as f:
                    f.write(resp.content)
                return output_path
            return None
        except:
            return None

    def delete_designed_voice(name):
        if not name:
            return "Please select a voice", list_designed_voices()
        try:
            resp = requests.delete(f"{API_URL}/voice_design/voices/{name}", timeout=30)
            if resp.status_code == 200:
                return f"Deleted '{name}'", list_designed_voices()
            return f"Error: {resp.text}", list_designed_voices()
        except Exception as e:
            return f"Error: {e}", list_designed_voices()

    preview_btn.click(
        preview_voice_design,
        inputs=[preview_desc, preview_lang, preview_text],
        outputs=[preview_audio, preview_status]
    )

    save_btn.click(
        save_designed_voice,
        inputs=[save_name, save_desc, save_lang],
        outputs=[save_status]
    )

    refresh_designed_btn.click(
        lambda: (gr.update(choices=list_designed_voices()), ""),
        outputs=[designed_voice_list, designed_voice_info]
    )

    designed_voice_list.change(
        get_designed_voice_info,
        inputs=[designed_voice_list],
        outputs=[designed_voice_info]
    )

    test_designed_btn.click(
        test_designed_voice,
        inputs=[designed_voice_list, test_designed_text],
        outputs=[test_designed_audio]
    )

    delete_designed_btn.click(
        delete_designed_voice,
        inputs=[designed_voice_list],
        outputs=[save_status, designed_voice_list]
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
