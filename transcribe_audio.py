import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description='Transcribe audio using a Whisper model')

parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Path to the fine-tuned Whisper model directory (checkpoint folder or final output dir)",
)
parser.add_argument(
    "--path_to_audio",
    type=str,
    required=True,
    help="Path to the audio file to be transcribed.",
)
parser.add_argument(
    "--language",
    type=str,
    default="hi",
    help="Language code (e.g., 'hi' for Hindi)",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="GPU device id (0) or -1 for CPU",
)

args = parser.parse_args()

transcribe = pipeline(
    task="automatic-speech-recognition",
    model=args.model_dir,
    chunk_length_s=30,
    device=args.device,
)

# Force language for decoding
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
    language=args.language,
    task="transcribe"
)

print("\nTranscription:\n")
print(transcribe(args.path_to_audio)["text"])
