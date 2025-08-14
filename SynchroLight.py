"""
SynchroLight/SpectraKeys
Audio-Reactive RGB Keyboard Controller for Lenovo LOQ 4-Zone Keyboard
======================================================================
Author: Abhayjeet Singh
Version: 1.0.0
Compatible with: Windows 10/11, Linux, Lenovo LOQ/Legion keyboards with 4-zone RGB

This script creates real-time audio-reactive lighting effects for your keyboard using MP3 files.
Based on: https://github.com/4JX/L5P-Keyboard-RGB
"""

import numpy as np
import librosa
import hid
import time
import threading
import queue
import struct
from collections import deque
from scipy import signal
from typing import Tuple, List, Optional
import colorsys
import json
import sys
import os
import argparse

# ==================== CONFIGURATION ====================

class Config:
    """Main configuration for the audio-reactive RGB controller"""
    
    def __init__(self):
        # Audio Settings
        self.SAMPLE_RATE = 22050  # Lower sample rate for better performance
        self.CHUNK_SIZE = 1024
        self.HOP_LENGTH = 512
        
        # RGB Settings
        self.FPS = 30  # Reduced for stability
        self.SMOOTHING_FACTOR = 0.8  # 0-1, higher = smoother transitions
        self.BRIGHTNESS_MULTIPLIER = 1.0  # Global brightness control
        
        # Frequency Bands (Hz) - optimized for 4 zones
        self.FREQ_BANDS = [
            (20, 250),      # Zone 1: Bass
            (250, 1000),    # Zone 2: Low-mids
            (1000, 4000),   # Zone 3: High-mids
            (4000, 11000)   # Zone 4: Highs (limited by sample rate)
        ]
        
        # Beat Detection
        self.BEAT_SENSITIVITY = 1.5
        self.BEAT_DECAY = 0.85
        
        # Color Schemes
        self.COLOR_MODE = "spectrum"
        self.CUSTOM_COLORS = [
            (255, 0, 0),    # Zone 1: Red
            (0, 255, 0),    # Zone 2: Green
            (0, 0, 255),    # Zone 3: Blue
            (255, 0, 255)   # Zone 4: Magenta
        ]
        
        # Effects
        self.ENABLE_BEAT_PULSE = True
        self.ENABLE_RIPPLE = True
        self.ENABLE_MOMENTUM = True
        self.MOMENTUM_WEIGHT = 0.3

# ==================== HID COMMUNICATION ====================

class LenovoRGBController:
    """Handles USB HID communication with Lenovo LOQ/Legion keyboards"""
    
    # Based on L5P-Keyboard-RGB protocol analysis
    VENDOR_ID = 0x048D
    PRODUCT_IDS = [
        0xC995, 0xC994, 0xC993,  # 2024 models
        0xC985, 0xC984, 0xC983,  # 2023 models  
        0xC975, 0xC973,          # 2022 models
        0xC965, 0xC963,          # 2021 models
        0xC955                   # 2020 models
    ]
    
    # Known device tuples (VID, PID, usage_page, usage)
    KNOWN_DEVICES = [
        (0x048d, 0xc995, 0xff89, 0x00cc), # 2024 Pro
        (0x048d, 0xc994, 0xff89, 0x00cc), # 2024
        (0x048d, 0xc993, 0xff89, 0x00cc), # 2024 LOQ
        (0x048d, 0xc985, 0xff89, 0x00cc), # 2023 Pro
        (0x048d, 0xc984, 0xff89, 0x00cc), # 2023
        (0x048d, 0xc983, 0xff89, 0x00cc), # 2023 LOQ
        (0x048d, 0xc975, 0xff89, 0x00cc), # 2022
        (0x048d, 0xc973, 0xff89, 0x00cc), # 2022 Ideapad
        (0x048d, 0xc965, 0xff89, 0x00cc), # 2021
        (0x048d, 0xc963, 0xff89, 0x00cc), # 2021 Ideapad
        (0x048d, 0xc955, 0xff89, 0x00cc), # 2020
    ]
    
    def __init__(self):
        self.device = None
        self.device_info = None
        self.connect()
        
    def connect(self):
        """Connect to the keyboard via HID using specific device matching"""
        print("Searching for Lenovo RGB keyboard...")
        
        # Find the correct device by matching usage page and usage
        matching_devices = []
        for device_info in hid.enumerate():
            if device_info['vendor_id'] == self.VENDOR_ID:
                device_tuple = (
                    device_info['vendor_id'],
                    device_info['product_id'],
                    device_info.get('usage_page', 0),
                    device_info.get('usage', 0)
                )
                
                # Check if this device matches our known devices
                if device_tuple in self.KNOWN_DEVICES:
                    matching_devices.append(device_info)
                    print(f"Found matching device: PID {device_info['product_id']:04X}, Usage: {device_info.get('usage_page', 0):04X}:{device_info.get('usage', 0):04X}")
        
        if not matching_devices:
            # Fallback: try any Lenovo device
            print("No exact match found, trying any Lenovo device...")
            for device_info in hid.enumerate():
                if device_info['vendor_id'] == self.VENDOR_ID:
                    matching_devices.append(device_info)
        
        # Try to connect to matching devices
        for device_info in matching_devices:
            try:
                self.device = hid.device()
                self.device.open_path(device_info['path'])
                self.device.set_nonblocking(1)
                self.device_info = device_info
                
                product_name = device_info.get('product_string', f"PID:{device_info['product_id']:04X}")
                print(f"✓ Connected to: {product_name}")
                
                # Test the connection with a simple command
                if self._test_connection():
                    return
                else:
                    print("Connection test failed, trying next device...")
                    self.device.close()
                    continue
                    
            except Exception as e:
                if self.device:
                    try:
                        self.device.close()
                    except:
                        pass
                continue
        
        raise Exception("❌ No compatible Lenovo RGB keyboard found!")
    
    def _test_connection(self) -> bool:
        """Test if the connection works by sending a simple command"""
        try:
            # Send a simple status command or null packet
            test_packet = [0xCC, 0x16, 0x01, 0x01, 0x01] + [0] * 28
            result = self.device.send_feature_report(bytes(test_packet))
            return result >= 0
        except:
            return False
    
    def set_colors(self, colors: List[Tuple[int, int, int]]):
        """
        Set RGB colors for all 4 zones using L5P-Keyboard-RGB protocol
        colors: List of 4 RGB tuples [(r,g,b), ...]
        """
        if not self.device:
            return
            
        if len(colors) != 4:
            raise ValueError("Must provide exactly 4 colors for 4 zones")
        
        try:
            # Build packet according to L5P-Keyboard-RGB source
            packet = bytearray(33)
            packet[0] = 0xCC   # Report ID
            packet[1] = 0x16   # Command 
            packet[2] = 0x01   # Effect: Static color
            packet[3] = 0x01   # Speed (1-4, doesn't matter for static)
            packet[4] = 0x02   # Brightness (1=low, 2=high)
            
            # RGB data starts at index 5, 3 bytes per zone
            for i, (r, g, b) in enumerate(colors):
                base_idx = 5 + (i * 3)
                packet[base_idx] = max(0, min(255, int(r)))
                packet[base_idx + 1] = max(0, min(255, int(g)))
                packet[base_idx + 2] = max(0, min(255, int(b)))
            
            # Try multiple communication methods
            success = False
            
            # Method 1: send_feature_report
            try:
                result = self.device.send_feature_report(packet)
                if result >= 0:
                    success = True
            except Exception as e:
                pass
            
            # Method 2: write (if feature report fails)
            if not success:
                try:
                    result = self.device.write(packet[1:])  # Skip report ID for write
                    if result >= 0:
                        success = True
                except Exception as e:
                    pass
            
            # Method 3: write with report ID
            if not success:
                try:
                    result = self.device.write(packet)
                    if result >= 0:
                        success = True
                except Exception as e:
                    pass
            
            if not success:
                # Only print error occasionally to avoid spam
                import random
                if random.random() < 0.1:  # 10% chance to print error
                    print("❌ All HID communication methods failed")
                
        except Exception as e:
            print(f"❌ HID error: {e}")
            # Try to reconnect on error
            try:
                self.device.close()
                self.connect()
            except:
                pass
    
    def set_effect(self, effect_type: int, speed: int = 2):
        """Set keyboard effect mode"""
        if not self.device:
            return
            
        try:
            packet = [0] * 33
            packet[0] = 0xCC
            packet[1] = 0x16
            packet[2] = effect_type  # 0x01=static, 0x03=breath, 0x04=wave, 0x06=smooth
            packet[3] = speed        # 1-4
            packet[4] = 0x02        # Brightness
            
            self.device.send_feature_report(bytes(packet))
            
        except Exception as e:
            print(f"Error setting effect: {e}")
    
    def close(self):
        """Close HID connection"""
        if self.device:
            try:
                # Reset to static white before closing
                self.set_colors([(255, 255, 255)] * 4)
                time.sleep(0.1)
                self.device.close()
                print("✓ Keyboard connection closed")
            except:
                pass

# ==================== AUDIO PROCESSING ====================

class MP3AudioProcessor:
    """Handles MP3 audio processing and frequency analysis"""
    
    def __init__(self, config: Config, mp3_file: str):
        self.config = config
        self.mp3_file = mp3_file
        self.audio_data = None
        self.sample_rate = config.SAMPLE_RATE
        self.current_position = 0
        self.total_frames = 0
        self.beat_history = deque(maxlen=20)
        self.last_beat_time = 0
        self.load_audio()
        
    def load_audio(self):
        """Load MP3 file using librosa"""
        try:
            print(f"Loading MP3: {self.mp3_file}")
            self.audio_data, loaded_sr = librosa.load(
                self.mp3_file, 
                sr=self.sample_rate,
                mono=True
            )
            self.total_frames = len(self.audio_data)
            duration = self.total_frames / self.sample_rate
            print(f"✓ Loaded {duration:.1f}s of audio at {self.sample_rate}Hz")
            
        except Exception as e:
            raise Exception(f"Failed to load MP3 file: {e}")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk"""
        if self.current_position >= self.total_frames:
            return None  # End of file
            
        end_pos = min(self.current_position + self.config.CHUNK_SIZE, self.total_frames)
        chunk = self.audio_data[self.current_position:end_pos]
        
        # Pad with zeros if chunk is too short
        if len(chunk) < self.config.CHUNK_SIZE:
            chunk = np.pad(chunk, (0, self.config.CHUNK_SIZE - len(chunk)))
            
        self.current_position += self.config.HOP_LENGTH
        return chunk
    
    def analyze_frequencies(self, audio_chunk: np.ndarray) -> List[float]:
        """
        Analyze audio chunk and return energy levels for each frequency band
        Returns: List of 4 normalized energy values (0-1)
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return [0.0] * 4
        
        # Apply window to reduce spectral leakage
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        
        # Compute FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), 1/self.sample_rate)
        
        # Calculate energy for each band
        band_energies = []
        for low_freq, high_freq in self.config.FREQ_BANDS:
            # Find frequency bin indices
            low_idx = np.searchsorted(freqs, low_freq)
            high_idx = np.searchsorted(freqs, high_freq)
            
            # Calculate average energy in band
            if high_idx > low_idx:
                band_energy = np.mean(fft[low_idx:high_idx])
            else:
                band_energy = 0
            
            band_energies.append(band_energy)
        
        # Normalize energies (with some smoothing to avoid division by zero)
        max_energy = max(band_energies) if max(band_energies) > 0.001 else 1
        normalized = [min(1.0, e / max_energy) for e in band_energies]
        
        return normalized
    
    def detect_beat(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple beat detection using energy-based onset detection
        Returns: True if beat detected
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False
        
        # Calculate instantaneous energy
        energy = np.sum(audio_chunk ** 2)
        self.beat_history.append(energy)
        
        if len(self.beat_history) < 5:
            return False
        
        # Check if current energy is significantly above recent average
        recent_avg = np.mean(list(self.beat_history)[-5:])
        longer_avg = np.mean(self.beat_history)
        
        if energy > recent_avg * self.config.BEAT_SENSITIVITY and energy > longer_avg * 1.2:
            current_time = time.time()
            # Debounce: minimum time between beats
            if current_time - self.last_beat_time > 0.1:
                self.last_beat_time = current_time
                return True
        
        return False
    
    def get_progress(self) -> float:
        """Get playback progress (0-1)"""
        if self.total_frames == 0:
            return 1.0
        return min(1.0, self.current_position / self.total_frames)
    
    def is_finished(self) -> bool:
        """Check if playback is finished"""
        return self.current_position >= self.total_frames

# ==================== VISUAL EFFECTS ====================

class EffectsEngine:
    """Manages visual effects and color transitions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_colors = [(0, 0, 0)] * 4
        self.target_colors = [(0, 0, 0)] * 4
        self.momentum_colors = [(0, 0, 0)] * 4
        self.beat_intensity = 0
        self.ripple_position = 0
        self.rainbow_offset = 0
        self.last_update = time.time()
        
    def update_colors(self, freq_energies: List[float], beat_detected: bool):
        """Update target colors based on frequency energies and effects"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Get base colors based on mode
        base_colors = self._get_base_colors(freq_energies)
        
        # Apply beat effect
        if self.config.ENABLE_BEAT_PULSE and beat_detected:
            self.beat_intensity = 1.0
        
        # Apply effects to each zone
        for i in range(4):
            r, g, b = base_colors[i]
            
            # Apply frequency-based brightness
            freq_brightness = freq_energies[i] * self.config.BRIGHTNESS_MULTIPLIER
            
            # Add beat pulse effect
            beat_boost = 0
            if self.beat_intensity > 0:
                beat_boost = self.beat_intensity * 0.4
            
            # Apply ripple effect
            ripple_boost = 0
            if self.config.ENABLE_RIPPLE and beat_detected:
                ripple_boost = self._calculate_ripple(i)
            
            # Combine brightness effects
            total_brightness = min(1.0, freq_brightness + beat_boost + ripple_boost)
            
            # Apply brightness to colors
            r = int(r * total_brightness)
            g = int(g * total_brightness)  
            b = int(b * total_brightness)
            
            # Apply momentum smoothing
            if self.config.ENABLE_MOMENTUM:
                prev_r, prev_g, prev_b = self.momentum_colors[i]
                weight = self.config.MOMENTUM_WEIGHT
                r = int(r * (1 - weight) + prev_r * weight)
                g = int(g * (1 - weight) + prev_g * weight)
                b = int(b * (1 - weight) + prev_b * weight)
                self.momentum_colors[i] = (r, g, b)
            
            self.target_colors[i] = (r, g, b)
        
        # Decay effects over time
        self.beat_intensity *= self.config.BEAT_DECAY
        
        # Update ripple animation
        if self.config.ENABLE_RIPPLE:
            self.ripple_position = (self.ripple_position + dt * 2) % 4
    
    def _get_base_colors(self, freq_energies: List[float]) -> List[Tuple[int, int, int]]:
        """Get base colors based on selected color mode"""
        if self.config.COLOR_MODE == "custom":
            return self.config.CUSTOM_COLORS.copy()
        
        elif self.config.COLOR_MODE == "spectrum":
            # Map each zone to different hue
            colors = []
            for i in range(4):
                hue = i / 4.0  # Spread across hue spectrum
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors.append(tuple(int(c * 255) for c in rgb))
            return colors
        
        elif self.config.COLOR_MODE == "rainbow":
            # Animated rainbow
            self.rainbow_offset = (self.rainbow_offset + 0.01) % 1.0
            colors = []
            for i in range(4):
                hue = (self.rainbow_offset + i / 4.0) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors.append(tuple(int(c * 255) for c in rgb))
            return colors
        
        elif self.config.COLOR_MODE == "fire":
            return [(255, 0, 0), (255, 100, 0), (255, 200, 0), (255, 255, 100)]
        
        elif self.config.COLOR_MODE == "ocean":
            return [(0, 0, 139), (0, 100, 200), (0, 200, 200), (100, 255, 200)]
        
        elif self.config.COLOR_MODE == "energy":
            # Color based on energy level of each band
            colors = []
            for energy in freq_energies:
                if energy < 0.3:
                    # Low energy: blue
                    colors.append((0, int(100 + energy * 300), int(200 + energy * 55)))
                elif energy < 0.7:
                    # Medium energy: green to yellow
                    colors.append((int(energy * 255), 255, 0))
                else:
                    # High energy: red
                    colors.append((255, int(255 * (1 - energy)), 0))
            return colors
        
        else:
            return [(255, 255, 255)] * 4  # Default white
    
    def _calculate_ripple(self, zone: int) -> float:
        """Calculate ripple brightness for a zone"""
        distance = abs(zone - self.ripple_position)
        # Create ripple effect
        if distance < 1:
            return (1 - distance) * 0.3
        elif distance < 2:
            return (2 - distance) * 0.1
        return 0
    
    def smooth_transition(self) -> List[Tuple[int, int, int]]:
        """Smoothly transition current colors toward target colors"""
        factor = self.config.SMOOTHING_FACTOR
        
        for i in range(4):
            curr_r, curr_g, curr_b = self.current_colors[i]
            targ_r, targ_g, targ_b = self.target_colors[i]
            
            # Exponential smoothing
            new_r = int(curr_r * factor + targ_r * (1 - factor))
            new_g = int(curr_g * factor + targ_g * (1 - factor))
            new_b = int(curr_b * factor + targ_b * (1 - factor))
            
            self.current_colors[i] = (new_r, new_g, new_b)
        
        return self.current_colors.copy()

# ==================== MAIN CONTROLLER ====================

class AudioRGBController:
    """Main controller that orchestrates everything"""
    
    def __init__(self, config: Config, mp3_file: str):
        self.config = config
        self.mp3_file = mp3_file
        self.rgb_controller = None
        self.audio_processor = None
        self.effects_engine = EffectsEngine(config)
        self.running = False
        
    def initialize(self):
        """Initialize all components"""
        print("Initializing Audio-Reactive RGB Controller...")
        
        # Initialize keyboard
        self.rgb_controller = LenovoRGBController()
        
        # Initialize audio processor
        self.audio_processor = MP3AudioProcessor(self.config, self.mp3_file)
        
        print("✓ All components initialized successfully!")
    
    def run(self):
        """Main execution loop"""
        if not self.rgb_controller or not self.audio_processor:
            raise Exception("Controller not initialized!")
        
        print(f"🎵 Starting audio-reactive RGB with: {os.path.basename(self.mp3_file)}")
        print(f"🌈 Color Mode: {self.config.COLOR_MODE}")
        print(f"⚡ FPS: {self.config.FPS}")
        print("\nPress Ctrl+C to stop\n")
        
        self.running = True
        frame_time = 1.0 / self.config.FPS
        frame_count = 0
        start_time = time.time()
        last_progress_update = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get next audio chunk
                audio_chunk = self.audio_processor.get_audio_chunk()
                
                if audio_chunk is None or self.audio_processor.is_finished():
                    print("\n🎵 Audio finished!")
                    break
                
                # Process audio
                freq_energies = self.audio_processor.analyze_frequencies(audio_chunk)
                beat_detected = self.audio_processor.detect_beat(audio_chunk)
                
                # Update visual effects
                self.effects_engine.update_colors(freq_energies, beat_detected)
                colors = self.effects_engine.smooth_transition()
                
                # Send to keyboard
                self.rgb_controller.set_colors(colors)
                
                # Progress display
                progress = self.audio_processor.get_progress()
                if progress - last_progress_update > 0.05:  # Update every 5%
                    progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                    print(f"\r🎵 [{progress_bar}] {progress*100:.1f}% ", end="", flush=True)
                    last_progress_update = progress
                
                # Beat indicator
                if beat_detected:
                    print("💥", end="", flush=True)
                
                # FPS limiting
                frame_count += 1
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
        finally:
            self.stop()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Stats: {frame_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} FPS)")
    
    def stop(self):
        """Stop the controller and clean up"""
        self.running = False
        print("\n🧹 Cleaning up...")
        
        if self.rgb_controller:
            # Fade out effect
            try:
                for brightness in range(10, -1, -1):
                    fade_colors = [(brightness * 25, brightness * 25, brightness * 25)] * 4
                    self.rgb_controller.set_colors(fade_colors)
                    time.sleep(0.05)
                
                self.rgb_controller.close()
            except:
                pass
        
        print("✓ Cleanup complete")

# ==================== UTILITY FUNCTIONS ====================

def test_keyboard():
    """Test keyboard connectivity and basic functionality"""
    print("🧪 Testing keyboard connection...")
    
    try:
        controller = LenovoRGBController()
        
        print("Testing color patterns...")
        test_patterns = [
            [(255, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],    # Zone 1 red
            [(0, 0, 0), (0, 255, 0), (0, 0, 0), (0, 0, 0)],    # Zone 2 green
            [(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0)],    # Zone 3 blue
            [(0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 255)],# Zone 4 white
            [(255, 128, 0)] * 4,                                # All orange
            [(0, 255, 255)] * 4,                                # All cyan
        ]
        
        for i, colors in enumerate(test_patterns):
            print(f"  Pattern {i+1}/6...")
            controller.set_colors(colors)
            time.sleep(0.8)
        
        # Reset to white
        controller.set_colors([(255, 255, 255)] * 4)
        time.sleep(0.5)
        controller.close()
        
        print("✅ Keyboard test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Keyboard test failed: {e}")
        return False

def debug_hid_devices():
    """Debug function to show all HID devices and their properties"""
    print("\n🔍 Debugging HID devices...")
    
    try:
        lenovo_devices = []
        all_devices = hid.enumerate()
        
        for device in all_devices:
            if device['vendor_id'] == 0x048D:
                lenovo_devices.append(device)
        
        if not lenovo_devices:
            print("❌ No Lenovo devices found!")
            print("\nAll devices:")
            for i, device in enumerate(all_devices[:10]):  # Show first 10
                print(f"  {i+1}. VID:{device['vendor_id']:04X} PID:{device['product_id']:04X} - {device.get('product_string', 'Unknown')}")
            return
        
        print(f"Found {len(lenovo_devices)} Lenovo device(s):")
        
        for i, device in enumerate(lenovo_devices):
            print(f"\n  Device {i+1}:")
            print(f"    Product: {device.get('product_string', 'Unknown')}")
            print(f"    VID: {device['vendor_id']:04X}")
            print(f"    PID: {device['product_id']:04X}")
            print(f"    Usage Page: {device.get('usage_page', 'N/A'):04X}")
            print(f"    Usage: {device.get('usage', 'N/A'):04X}")
            print(f"    Path: {device['path']}")
            
            # Check if it matches known devices
            device_tuple = (
                device['vendor_id'],
                device['product_id'],
                device.get('usage_page', 0),
                device.get('usage', 0)
            )
            
            known_match = device_tuple in LenovoRGBController.KNOWN_DEVICES
            print(f"    Known device: {'✅ Yes' if known_match else '❌ No'}")
            
            # Try to open and test
            try:
                test_device = hid.device()
                test_device.open_path(device['path'])
                test_device.set_nonblocking(1)
                
                # Try to send a test packet
                test_packet = [0xCC, 0x16, 0x01, 0x01, 0x01] + [0] * 28
                result = test_device.send_feature_report(bytes(test_packet))
                print(f"    Test result: {result}")
                
                test_device.close()
                
            except Exception as e:
                print(f"    Test failed: {e}")
    
    except Exception as e:
        print(f"❌ Error during HID debug: {e}")

def list_supported_keyboards():
    """List all connected Lenovo devices"""
    print("\n🔍 Scanning for Lenovo devices...")
    
    try:
        found_devices = []
        for device_info in hid.enumerate():
            if device_info['vendor_id'] == 0x048D:
                found_devices.append({
                    'product_id': device_info['product_id'],
                    'product_string': device_info.get('product_string', 'Unknown'),
                    'path': device_info['path']
                })
        
        if found_devices:
            print(f"Found {len(found_devices)} Lenovo device(s):")
            for i, device in enumerate(found_devices):
                print(f"  {i+1}. {device['product_string']} (PID: {device['product_id']:04X})")
        else:
            print("❌ No Lenovo devices found")
            
        return found_devices
        
    except Exception as e:
        print(f"❌ Error scanning devices: {e}")
        return []

def validate_mp3_file(file_path: str) -> bool:
    """Validate that the MP3 file exists and can be loaded"""
    if not os.path.exists(file_path):
        print(f"❌ MP3 file not found: {file_path}")
        return False
    
    try:
        # Test load a small portion
        test_audio, sr = librosa.load(file_path, duration=1.0, sr=22050)
        duration = librosa.get_duration(filename=file_path)
        print(f"✅ MP3 file valid: {os.path.basename(file_path)} ({duration:.1f}s)")
        return True
    except Exception as e:
        print(f"❌ Invalid MP3 file: {e}")
        return False

# ==================== MAIN ENTRY POINT ====================

def main():
    parser = argparse.ArgumentParser(description='Audio-Reactive RGB Controller for Lenovo Keyboards')
    parser.add_argument('mp3_file', nargs='?', help='Path to MP3 file')
    parser.add_argument('--test', action='store_true', help='Test keyboard connection')
    parser.add_argument('--debug', action='store_true', help='Debug HID devices') 
    parser.add_argument('--list', action='store_true', help='List connected devices') 
    parser.add_argument('--color-mode', choices=['spectrum', 'rainbow', 'fire', 'ocean', 'energy', 'custom'], 
                       default='spectrum', help='Color scheme to use')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (10-60)')
    parser.add_argument('--brightness', type=float, default=1.0, help='Brightness multiplier (0.1-2.0)')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════╗
║       Audio-Reactive RGB Controller v2.0             ║
║           For Lenovo LOQ/Legion Keyboards             ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    if args.debug:
        debug_hid_devices()
        return
    
    if args.test:
        test_keyboard()
        return
        
    if args.list:
        list_supported_keyboards()
        return
    
    if not args.mp3_file:
        print("❌ Please provide an MP3 file path")
        print("Usage: python SynchroLight.py <mp3_file>")
        print("       python SynchroLight.py --test       # Test keyboard")
        print("       python SynchroLight.py --debug      # Debug HID devices")
        print("Example: python SynchroLight.py music.mp3")
        return
    
    # Validate inputs
    if not validate_mp3_file(args.mp3_file):
        return
    
    # Create config
    config = Config()
    config.COLOR_MODE = args.color_mode
    config.FPS = max(10, min(60, args.fps))
    config.BRIGHTNESS_MULTIPLIER = max(0.1, min(2.0, args.brightness))
    
    # Run controller
    try:
        controller = AudioRGBController(config, args.mp3_file)
        controller.initialize()
        controller.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())