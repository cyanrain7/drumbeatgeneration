# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imports objects from music modules into the top-level music namespace."""

from music.abc_parser import parse_abc_tunebook
from music.abc_parser import parse_abc_tunebook_file

from music.chord_inference import ChordInferenceError
from music.chord_inference import infer_chords_for_sequence

from music.chord_symbols_lib import chord_symbol_bass
from music.chord_symbols_lib import chord_symbol_pitches
from music.chord_symbols_lib import chord_symbol_quality
from music.chord_symbols_lib import chord_symbol_root
from music.chord_symbols_lib import ChordSymbolError
from music.chord_symbols_lib import pitches_to_chord_symbol
from music.chord_symbols_lib import transpose_chord_symbol

from music.chords_encoder_decoder import ChordEncodingError
from music.chords_encoder_decoder import MajorMinorChordOneHotEncoding
from music.chords_encoder_decoder import PitchChordsEncoderDecoder
from music.chords_encoder_decoder import TriadChordOneHotEncoding

from music.chords_lib import BasicChordRenderer
from music.chords_lib import ChordProgression

from music.constants import *  # pylint: disable=wildcard-import

from music.drums_encoder_decoder import MultiDrumOneHotEncoding

from music.drums_lib import DrumTrack
from music.drums_lib import midi_file_to_drum_track

from music.encoder_decoder import ConditionalEventSequenceEncoderDecoder
from music.encoder_decoder import EventSequenceEncoderDecoder
from music.encoder_decoder import LookbackEventSequenceEncoderDecoder
from music.encoder_decoder import MultipleEventSequenceEncoder
from music.encoder_decoder import OneHotEncoding
from music.encoder_decoder import OneHotEventSequenceEncoderDecoder
from music.encoder_decoder import OneHotIndexEventSequenceEncoderDecoder
from music.encoder_decoder import OptionalEventSequenceEncoder

from music.events_lib import NonIntegerStepsPerBarError

from music.lead_sheets_lib import LeadSheet

from music.melodies_lib import BadNoteError
from music.melodies_lib import Melody
from music.melodies_lib import midi_file_to_melody
from music.melodies_lib import PolyphonicMelodyError

from music.melody_encoder_decoder import KeyMelodyEncoderDecoder
from music.melody_encoder_decoder import MelodyOneHotEncoding

from music.melody_inference import infer_melody_for_sequence
from music.melody_inference import MelodyInferenceError

from music.midi_io import midi_file_to_note_sequence
from music.midi_io import midi_file_to_sequence_proto
from music.midi_io import midi_to_note_sequence
from music.midi_io import midi_to_sequence_proto
from music.midi_io import MIDIConversionError
from music.midi_io import sequence_proto_to_midi_file
from music.midi_io import sequence_proto_to_pretty_midi

from music.midi_synth import fluidsynth
from music.midi_synth import synthesize

from music.model import BaseModel

from music.musicxml_parser import MusicXMLDocument
from music.musicxml_parser import MusicXMLParseError

from music.musicxml_reader import musicxml_file_to_sequence_proto
from music.musicxml_reader import musicxml_to_sequence_proto
from music.musicxml_reader import MusicXMLConversionError

from music.notebook_utils import play_sequence
from music.notebook_utils import plot_sequence

from music.performance_controls import all_performance_control_signals
from music.performance_controls import NoteDensityPerformanceControlSignal
from music.performance_controls import PitchHistogramPerformanceControlSignal

from music.performance_encoder_decoder import ModuloPerformanceEventSequenceEncoderDecoder
from music.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder
from music.performance_encoder_decoder import PerformanceModuloEncoding
from music.performance_encoder_decoder import PerformanceOneHotEncoding

from music.performance_lib import MetricPerformance
from music.performance_lib import Performance

from music.pianoroll_encoder_decoder import PianorollEncoderDecoder

from music.pianoroll_lib import PianorollSequence

from music.sequences_lib import apply_sustain_control_changes
from music.sequences_lib import BadTimeSignatureError
from music.sequences_lib import concatenate_sequences
from music.sequences_lib import extract_subsequence
from music.sequences_lib import infer_dense_chords_for_sequence
from music.sequences_lib import MultipleTempoError
from music.sequences_lib import MultipleTimeSignatureError
from music.sequences_lib import NegativeTimeError
from music.sequences_lib import quantize_note_sequence
from music.sequences_lib import quantize_note_sequence_absolute
from music.sequences_lib import quantize_to_step
from music.sequences_lib import steps_per_bar_in_quantized_sequence
from music.sequences_lib import steps_per_quarter_to_steps_per_second
from music.sequences_lib import trim_note_sequence
