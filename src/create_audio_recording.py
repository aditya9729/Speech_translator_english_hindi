import pyaudio
import wave
import os



def record_sound(WAVE_OUTPUT_FILENAME,FORMAT,CHANNELS,RATE,CHUNK):
    """Records sound through microphone and saves it as a wav file.
    :param WAVE_OUTPUT_FILENAME: Input wav file to the model
    :param FORMAT: Pyint16 format
    :param CHANNELS: Number of channels - mono
    :param RATE: Sampling rate
    :param CHUNK: Byte Chunk
    :return: None
    """
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(os.path.join('static',WAVE_OUTPUT_FILENAME), 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__=='__main__':
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()

    WAVE_OUTPUT_FILENAME = "input.wav"

    record_sound(WAVE_OUTPUT_FILENAME,FORMAT, CHANNELS, RATE, CHUNK)


