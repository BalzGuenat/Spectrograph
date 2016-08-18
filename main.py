import pyaudio
import pyglet
import numpy as np
import matplotlib.pyplot as plt
import time

### 'Physical' dimensions:   ###

margin = 50

dot_h = 5
dot_w = 20
sep_h = 4
sep_w = 10
num_levels = 32
num_bands = 24

### Aesthetic configuration: ###

color_low = (0,255,0)
color_mid = (255,255,0)
color_high = (255,0,0)

thresh_mid = .5
thresh_high = .9

def draw_border():
    pos_y = margin - 2 * sep_h + 50
    pos_x = margin - sep_w
    d_x = num_bands * (dot_w + sep_w) + sep_w
    d_y = num_levels * (dot_h + sep_h) + 2 * sep_h
    pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
        ('v2i', (pos_x, pos_y,
             pos_x, pos_y+d_y,
             pos_x+d_x, pos_y+d_y,
             pos_x+d_x, pos_y))
    )


################################

levels = np.zeros(num_bands).astype(int)
holds = np.zeros(num_bands).astype(int)
input = np.zeros(num_bands)
max_sensitivity = 1000000
# max_sensitivity = 0
max_input = max_sensitivity
switch = True

thresh_mid = int((thresh_mid * num_levels))
thresh_high = int((thresh_high * num_levels))


buffer_size = 2**10 # 1024
max_freq = 8000
min_freq = 64
rate = 2 * max_freq
frame_time = 1/rate

lo = np.log(min_freq)
hi = np.log(max_freq)
step = (hi-lo) / (num_bands-1)
print([np.exp(lo+i*step) for i in range(num_bands)])

band_freqs = [np.exp(lo+i*step) for i in range(num_bands)]

ap = pyaudio.PyAudio()
stream = ap.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=buffer_size)
xs_buffer = np.arange(buffer_size) * frame_time
xs = np.arange(buffer_size) * frame_time
audio = np.empty(buffer_size, dtype=np.int16)

DEBUG_PLOT = False
if DEBUG_PLOT:
    fig = plt.figure()
    fig.show()

def update(dt):
    global levels
    global holds
    global input
    global max_input, max_sensitivity
    global fig
    global band_freqs
    global switch

    # print(dt)

    audio_string = stream.read(buffer_size)
    audio = np.fromstring(audio_string, dtype=np.int16)
    data = audio.flatten()
    ys = np.fft.rfft(data)
    ys = np.absolute(ys)
    freqs = np.fft.rfftfreq(buffer_size, frame_time)

    if DEBUG_PLOT:
        fig.gca().clear()
        fig.gca().plot(freqs, ys)
        fig.canvas.draw()

    input = np.zeros(num_bands)
    i = 1
    for j in range(num_bands):
        sum = 0
        num = 0
        m = 0
        while i < len(freqs) and freqs[i] < band_freqs[j]:
            sum += ys[i]
            num += 1
            m = max(m,ys[i])
            i += 1
        input[j] = m
        # print(i)

    max_sens = max_sensitivity
    # max_sens = 2.2 ** np.log(max_sensitivity)
    # input = 2.2 ** np.log(input)
    max_sens = np.log(max_sens)
    input = np.log(input)

    max_sens -= 10
    input -= 10

    a = .1
    max_input = a * input.max() + (1-a) * max_input
    input = input / max(max_input, max_sens) * .9

    # print((input.max(), max(max_input, max_sens)))

    input = np.clip(input, 0, 1)

    # a = .4
    # input = a * input + (1.-a) * np.random.random(num_bands)
    # input = np.ones(num_bands) * 1.

    levels = (input * num_levels).astype(int)
    switch = not switch
    if switch or True:
        holds = np.maximum(holds-1, levels)
    else:
        holds = np.maximum(holds, levels)

window = pyglet.window.Window(1000,600)

@window.event
def on_draw():
    window.clear()
    draw_border()

    for band in range(num_bands):
        pos_x = margin + band * (dot_w + sep_w)
        for level in range(levels[band]):
            pos_y = margin + level * (dot_h + sep_h) + 50
            if level < thresh_mid:
                color = color_low
            elif level < thresh_high:
                color = color_mid
            else:
                color = color_high
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                ('v2i', (pos_x, pos_y,
                     pos_x, pos_y+dot_h,
                     pos_x+dot_w, pos_y+dot_h,
                     pos_x+dot_w, pos_y)),
                ('c3B', 4*color)
            )

        hold = holds[band]
        if (hold > 0):
            pos_y = margin + (hold-1) * (dot_h + sep_h) + 50
            if hold < thresh_mid:
                color = color_low
            elif hold < thresh_high:
                color = color_mid
            else:
                color = color_high
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                ('v2i', (pos_x, pos_y,
                     pos_x, pos_y+dot_h,
                     pos_x+dot_w, pos_y+dot_h,
                     pos_x+dot_w, pos_y)),
                ('c3B', 4*color)
            )


        # label = pyglet.text.Label("{0:.0f}".format(band_freqs[band]),
        #                           font_name='Arial',
        #                           font_size=12,
        #                           x=pos_x, y=20)
        # label.draw()


pyglet.clock.schedule_interval(update, 1/30)
pyglet.app.run()

ap.close(stream)
