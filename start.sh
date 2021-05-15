#!/bin/bash

git -C TTS fetch
if [ "$(git -C TTS diff pt-br origin/pt-br)" ]; then
	git -C TTS pull origin pt-br
fi

if [ $UPDATE_MODEL ]; then
	gdown --id $MODEL_LINK -O tts.zip
	unzip -o tts.zip
	mv tts/* /app/TTS/models
	rm -rf tts tts.zip
	rm -rf /app/voice_response || true
fi

uvicorn main:app --host 0.0.0.0 --port 5025
