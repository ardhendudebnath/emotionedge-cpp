"""EmotionEdge offline ML factory (blueprint "Offline ML factory · Python").

Trains, compresses and exports the models the C++ runtime loads. Nothing in this package is
shipped to the device: the runtime has no Python dependency.
"""

__all__ = ["emotion_space", "metrics"]
