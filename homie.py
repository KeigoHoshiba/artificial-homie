import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import threading
import queue
import time
from matplotlib.animation import FuncAnimation

# マイク入力の設定
CHUNK = 4096*2  # バッファサイズ
FORMAT = pyaudio.paInt16  # 音声フォーマット
CHANNELS = 1  # モノラル
RATE = 48000  # サンプリングレート
WIDTH = 200  # フォルマントの周波数範囲
GAIN = 6  # ブーストするゲイン
FADE = 300 # 出力のチャンク間クロスフェード

# キューの設定
audio_queue = queue.Queue()
plot_queue = queue.Queue()

p = pyaudio.PyAudio()

is_open = True
def on_close(event):
    global is_open
    is_open = False

# プロットの設定
fig, ax = plt.subplots()
fig.canvas.mpl_connect("close_event", on_close)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.set_ylim([100, 10**8])
ax.set_xscale("log")
ax.set_yscale("log")
freq = np.fft.fftfreq(CHUNK, d=1 / RATE)
line_input, = ax.plot(freq[:CHUNK // 2], np.zeros(CHUNK // 2), linewidth=0.5, label="Input")
line_output, = ax.plot(freq[:CHUNK // 2], np.zeros(CHUNK // 2), linewidth=0.5, label="Output")
ax.legend()

# ストリームのコールバック関数
def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# ストリームの設定
stream_input = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK,
                      stream_callback=callback)

stream_output = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True)

def process_audio():
    prev_boosted_samples = np.zeros(FADE)
    while is_open:
        if not audio_queue.empty():
            data = audio_queue.get()
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # FFTを実行
            fft_data = np.fft.rfft(samples)
            freqs = np.fft.rfftfreq(len(samples), 1.0 / RATE)

            # 第二フォルマントの解析
            snd = parselmouth.Sound(samples, sampling_frequency=RATE)
            formant = call(snd, "To Formant (burg)", 0.025, 5, 5500, 0.025, 50)
            second_formant_freq = call(formant, "Get value at time", 2, 0.1, 'Hertz', 'Linear')
            print(f"{int(second_formant_freq):4}")

            # ブースト
            boost_range = (freqs > (second_formant_freq - WIDTH)) & (freqs < (second_formant_freq + WIDTH))
            boosted_fft = np.zeros_like(fft_data)
            boosted_fft[boost_range] = fft_data[boost_range] * GAIN

            # 逆FFT
            boosted_samples = np.fft.irfft(boosted_fft, n=CHUNK+FADE)
            # クロスフェード
            mask = np.arange(0, 1, 1/float(FADE))
            boosted_samples[:FADE] = (1 - mask)*prev_boosted_samples + mask*boosted_samples[:FADE]
            stream_output.write(boosted_samples[:-FADE].astype(np.int16).tobytes())
            prev_boosted_samples = boosted_samples[-FADE:]

            # プロットデータをキューに追加
            plot_queue.put((freqs, fft_data, boosted_fft))

def update_plot(frame):
    if not plot_queue.empty():
        freqs, fft_data, boosted_fft = plot_queue.get()
        line_input.set_data(freqs, np.abs(fft_data))
        line_output.set_data(freqs, np.abs(boosted_fft))
        # print(np.abs(boosted_fft))
    return line_input, line_output

# スレッドの起動
process_thread = threading.Thread(target=process_audio)
process_thread.start()

# アニメーションの設定
ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

# ストリームを開始
stream_input.start_stream()
stream_output.start_stream()

plt.show()

# ストリームを閉じる
stream_input.stop_stream()
stream_input.close()
stream_output.stop_stream()
stream_output.close()
p.terminate()