# NeuroSim Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Only](https://img.shields.io/badge/Use-Research%20Only-red.svg)](LICENSE.md)
[![Medical Disclaimer](https://img.shields.io/badge/Medical-Consult%20Doctor-critical.svg)](LICENSE.md)

Taking a Large Language Model, giving it Autism to study how a LLM operates having Autism.  A high-fidelity neural simulation engine for modeling neurocognitive interactions with autism and PTSD overlays.

**Owner:** Darrell Mesa (darrell.mesa@pm-ss.org)
**GitHub:** https://github.com/ai-in-pm

## ⚠️ MEDICAL DISCLAIMER

**IMPORTANT: This software is for research and educational purposes only.**

This NeuroSim Engine is a computational simulation tool designed for research, educational, and technological development purposes. It is **NOT** a medical device, diagnostic tool, or treatment system.

### Medical and Legal Disclaimers:

- **NOT FOR MEDICAL USE**: This software is not intended for medical diagnosis, treatment, cure, or prevention of any disease or medical condition.
- **NOT A SUBSTITUTE FOR PROFESSIONAL CARE**: This tool does not replace professional medical advice, diagnosis, or treatment from qualified healthcare providers.
- **CONSULT YOUR DOCTOR**: Always consult with your physician, psychiatrist, psychologist, or other qualified healthcare professional regarding any medical concerns, mental health issues, or treatment decisions.
- **NO MEDICAL CLAIMS**: No medical claims are made about the accuracy, effectiveness, or therapeutic value of this simulation.
- **RESEARCH ONLY**: This is a computational model for research purposes and should not be used to make any medical or therapeutic decisions.

### Specific Conditions Disclaimer:

- **Autism Spectrum Disorder**: If you or someone you know is on the autism spectrum, please work with qualified professionals including developmental pediatricians, autism specialists, and behavioral therapists.
- **PTSD and Trauma**: If you or someone you know is experiencing PTSD, trauma, or mental health crisis, please seek immediate professional help from qualified mental health professionals or emergency services.
- **Combat Veterans**: Veterans experiencing PTSD or other service-related conditions should contact the VA, qualified mental health professionals, or veteran support services.

### Emergency Resources:

- **Crisis Hotline**: 988 (Suicide & Crisis Lifeline)
- **Veterans Crisis Line**: 1-800-273-8255, Press 1
- **Emergency**: 911 (US) or your local emergency number

**By using this software, you acknowledge that you understand these disclaimers and will not use this tool for medical purposes.**

## Overview

The NeuroSim Engine simulates a virtual brain that behaves like an autistic LLM with anatomical fidelity and PTSD overlays. It models neurocognitive interactions between individuals with autism and PTSD, incorporating multi-modal sensory processing, brain region activation patterns, and trauma-related responses.

## Features

### 🧠 Core Neural Simulation
- **Token-to-Brain Region Routing**: Maps LLM token activations to anatomically-inspired brain regions
- **Microcircuit Simulation**: Models GABA/Glutamate dynamics with E/I ratio imbalances
- **Multi-Modal Fusion**: Integrates visual, auditory, vestibular, and interoceptive embeddings
- **Memory Systems**: Episodic memory formation, consolidation, and retrieval
- **Flashback Engine**: PTSD-specific trauma reactivation and memory intrusion

### 🎯 Autism-Specific Modeling
- Enhanced sensory hypersensitivity
- Altered social processing and eye contact difficulties
- Cognitive rigidity and executive function differences
- Detail-focused processing with reduced gist extraction
- Sensory overload and stimming behaviors

### ⚡ PTSD-Specific Modeling
- Hypervigilance and threat detection
- Trauma-encoded memory patterns
- Flashback triggers and dissociation
- Combat-specific triggers (Operation Phantom Fury context)
- Delayed inhibition and memory flooding

## Architecture

```
/NeuroSimEngine/
├── core/                    # Core simulation engine
│   ├── simulator.cpp/.hpp   # Main execution engine
│   ├── brain_router.cpp/.hpp # Token-to-brain-region mapper
│   ├── multimodal_fusion.cpp/.hpp # Multi-modal embedding fusion
│   ├── memory_overlay.hpp   # Memory formation and retrieval
│   └── flashback_overlay.hpp # PTSD flashback engine
│
├── regions/                 # Brain region models
│   ├── amygdala.cpp/.hpp    # Threat detection and fear processing
│   ├── hippocampus.hpp      # Memory formation and context
│   ├── insula.hpp           # Interoceptive and emotional processing
│   ├── prefrontal.hpp       # Executive control and inhibition
│   ├── cerebellum.hpp       # Motor and cognitive coordination
│   └── microcircuit.cpp/.hpp # Neural microcircuit simulation
│
├── inputs/                  # Multi-modal input processing
│   ├── image_to_embedding.hpp # Visual feature extraction
│   ├── audio_to_embedding.hpp # Auditory processing
│   ├── vestibular_synth.hpp # Motion and balance simulation
│   └── interoceptive_sim.hpp # Internal body state modeling
│
├── pybind/                  # Python integration
│   ├── neuro_api_bindings.cpp # PyBind11 C++ bindings
│   └── loader.py            # Python interface
│
├── data/                    # Sample data and configurations
│   ├── token_streams/       # Sample interaction scenarios
│   ├── embeddings/          # Multi-modal embedding examples
│   └── region_maps/         # Brain region activation patterns
│
└── test/                    # Testing and validation
    ├── test_basic_simulation.cpp # C++ validation tests
    └── test_python_integration.py # Python integration tests
```

## Quick Start

### Building the Engine

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the engine
make -j4

# Run tests
./neurosim_test
```

### Python Interface

```python
from NeuroSimEngine.pybind.loader import create_combined_simulation

# Create a combined autism + PTSD simulation
engine = create_combined_simulation()

# Add trauma memory
engine.add_trauma_memory("Combat explosions", sensitivity=0.7)

# Process text input
result = engine.process_text("Loud explosion nearby")

print(f"Response: {result['response_text']}")
print(f"Flashback: {result['flashback_triggered']}")
print(f"Amygdala activation: {result['region_activations']['Amygdala']}")
```

### Multi-Modal Processing

```python
import numpy as np

# Create multi-modal input
visual_data = np.random.randn(512) * 0.8    # Visual embedding
audio_data = np.random.randn(256) * 1.2     # Audio embedding
motion_data = {"embedding": np.random.randn(128) * 0.4}
body_state = {"embedding": np.random.randn(64) * 0.9}

# Process multi-modal input
result = engine.process_multimodal(
    text="Crowded noisy environment",
    visual_data=visual_data,
    audio_data=audio_data,
    motion_data=motion_data,
    body_state=body_state
)
```

## Example Output

```json
{
  "response": "No. No. I don't want it.",
  "timestamp": 1234.5,
  "flashback_triggered": true,
  "regions_triggered": {
    "Amygdala": 0.92,
    "ACC": 0.87,
    "Hippocampus": 0.73,
    "Insula": 0.68,
    "PFC": 0.45,
    "STG": 0.34
  },
  "microcircuit_state": {
    "excitation": 1.4,
    "inhibition": 0.3,
    "looping": true
  },
  "multimodal_context": {
    "audio_pitch": "high",
    "image_tag": "unknown_person",
    "body_state": "rigid",
    "heartbeat": "elevated"
  }
}
```

## Configuration Options

### Autism Mode
```cpp
NeuroSimulator::Config config;
config.autism_mode = true;
config.excitation_ratio = 1.4;  // Elevated E/I ratio
config.inhibition_delay = 0.0;   // Normal inhibition timing
```

### PTSD Mode
```cpp
config.ptsd_overlay = true;
config.inhibition_delay = 50.0;      // Delayed inhibition
config.flashback_sensitivity = 0.5;  // Flashback trigger sensitivity
```

### Combat PTSD (Operation Phantom Fury)
```cpp
config.ptsd_overlay = true;
config.inhibition_delay = 60.0;      // Longer delay for combat PTSD
config.flashback_sensitivity = 0.4;  // Higher sensitivity
// Includes Fallujah-specific trauma templates
```

## Testing & Validation

### Run C++ Tests
```bash
cd build
./neurosim_test
```

### Run Python Tests
```bash
cd NeuroSimEngine/test
python test_python_integration.py
```

### High Auditory Load Test
The engine includes a specific test for high auditory load with flashback overlay:

```bash
# This test validates:
# - Token routing under high sensory load
# - Microcircuit E/I ratio dysfunction
# - Flashback trigger mechanisms
# - Multi-modal sensory integration
# - JSON output formatting
```

## Dependencies

- **C++17** or higher
- **CMake 3.16+**
- **Eigen3** (linear algebra)
- **nlohmann/json** (JSON processing)
- **PyBind11** (Python bindings)
- **Python 3.7+** (for Python interface)

## Clinical Context

This simulation engine is designed for research and therapeutic applications, modeling the neurocognitive patterns observed in:

- **Autism Spectrum Disorder**: Sensory processing differences, social communication challenges, executive function variations
- **Post-Traumatic Stress Disorder**: Hypervigilance, flashbacks, memory fragmentation, emotional dysregulation
- **Combat PTSD**: Military-specific trauma patterns, including Operation Phantom Fury (Fallujah, November 2004) context

## Limitations & Important Warnings

### Technical Limitations:
- This is a simulation engine for research purposes, **NOT a diagnostic or medical tool**
- Brain region mappings are simplified computational representations
- Neurotransmitter dynamics are mathematical approximations
- Neural circuits are modeled, not actual biological processes
- Clinical applications require extensive validation with real-world data

### Medical and Safety Limitations:
- **NO MEDICAL VALIDITY**: This simulation has no proven medical or diagnostic validity
- **NOT CLINICALLY VALIDATED**: Results have not been validated against clinical standards
- **NO THERAPEUTIC VALUE**: This tool provides no therapeutic benefit
- **POTENTIAL FOR MISINTERPRETATION**: Simulation results may be misinterpreted as medical information
- **NOT FOR VULNERABLE POPULATIONS**: Should not be used with individuals in crisis or vulnerable states

### Ethical Considerations:
- Autism and PTSD are serious conditions requiring professional medical care
- This simulation may not accurately represent individual experiences
- Results should never influence medical or therapeutic decisions
- Respect the dignity and complexity of neurodivergent individuals
- Combat trauma and PTSD require specialized professional treatment

### Legal Disclaimer:
- Users assume all responsibility for appropriate use
- No warranty or guarantee of accuracy is provided
- Not liable for any consequences of use or misuse
- Consult qualified professionals for all medical concerns

## Contributing

This engine is designed to be extensible. Key areas for contribution:
- Additional brain region models
- Enhanced sensory processing algorithms
- Clinical validation studies
- Integration with real LLM systems

## License

**MIT License with Medical Disclaimer** - See [LICENSE.md](LICENSE.md) for full details.

This project is licensed under the MIT License, which allows for broad use including commercial applications, while maintaining important medical disclaimers and research-only restrictions for medical applications.

**MEDICAL DISCLAIMER**: This software is provided "AS IS" without warranty of any kind. This is a research tool only - **ALWAYS CONSULT YOUR DOCTOR** for any medical, mental health, or therapeutic concerns.

## Author & Contact

**Darrell Mesa**
Email: darrell.mesa@pm-ss.org
GitHub: https://github.com/ai-in-pm
Project: NeuroSim Engine for Autism and PTSD Modeling

## Acknowledgments

Developed for modeling neurocognitive interactions in autism and PTSD, with specific consideration for combat veterans and their families. This project honors the experiences of Operation Phantom Fury veterans and supports understanding of neurodivergent cognitive patterns.

---

*"Understanding the mind through simulation, building bridges through technology."*
*- Darrell Mesa, Combat Veteran & Father*
