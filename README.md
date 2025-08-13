# SpectraKeys

**Audio-Reactive RGB Controller for Lenovo LOQ/Legion Keyboards**

SpectraKeys is a Python project I built to make my Lenovo LOQ keyboard light up in sync with music. It listens to an MP3 file, analyzes it in real time, and maps different frequency ranges to the four RGB lighting zones on the keyboard.

I wanted something that combined audio processing, visual effects, and direct hardware control. This project was my way of exploring USB HID communication, FFT-based frequency analysis, and beat detection, all while creating something I actually use on my own setup.

---

## Features

* Real-time frequency analysis using FFT
* Four distinct lighting zones mapped to bass, low-mids, high-mids, and highs
* Beat detection for extra visual effects like pulses and ripples
* Configurable color modes (spectrum, rainbow, energy, and more)
* Adjustable brightness, FPS, and smoothing for smooth transitions
* Written entirely in Python for easy modification and learning

---

## How It Works

1. **Audio Input**
   Loads an MP3 file and processes it chunk-by-chunk using `librosa`.
2. **Frequency Analysis**
   Splits the audio into four frequency bands, each mapped to a keyboard zone.
3. **Beat Detection**
   Checks for spikes in energy to trigger extra lighting effects.
4. **HID Communication**
   Sends RGB color data directly to the keyboard using USB HID packets.

---

## Installation

**Requirements**:

* Python 3.8+
* Lenovo LOQ or Legion keyboard with 4-zone RGB
* Windows 10/11 or Linux
* MP3 file to test with

**Install dependencies:**

```bash
pip install numpy librosa hidapi scipy
```

---

## Usage

**Usage**: Debug your HID setup
```bash
python rgb_controller.py --debug
```
**Test keyboard connection**  
```bash
python rgb_controller.py --test
```
**Run with MP3**
```bash
python rgb_controller.py song.mp3
```
**Try this first:**
```bash
python rgb_controller.py --debug
```
---

## Future Plans

* Play the audio file through the system while running the lights
* Add Windows loopback capture so it reacts to system audio
* Microphone input for live shows
* More visual effects like wave patterns or strobes

---

## License

This project is released under the **Apache License 2.0** so anyone can use, modify, and share it with attribution.

---

## Why I Built This

I’ve always liked projects that combine code with something physical you can see and enjoy. Writing SpectraKeys taught me about audio signal processing, timing in real-time applications, and low-level hardware protocols. Plus, it’s just fun to watch my keyboard dance to my music.

---
