#!/usr/bin/env python3
"""
NeuroSim Engine Python Interface
Python bridge to the C++ NeuroSim Engine for neural simulation

⚠️ MEDICAL DISCLAIMER: FOR RESEARCH ONLY
This software is for research and educational purposes only.
NOT for medical diagnosis or treatment. Always consult your doctor.

This module provides a high-level Python interface to the NeuroSim Engine,
enabling easy integration with Python-based LLM systems and data analysis tools.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    import neurosim_py
    NEUROSIM_AVAILABLE = True
except ImportError:
    NEUROSIM_AVAILABLE = False
    logging.warning("NeuroSim C++ module not available. Using simulation mode.")

class NeuroSimEngine:
    """
    High-level Python interface to the NeuroSim Engine
    
    This class provides a convenient Python wrapper around the C++ NeuroSim Engine,
    with additional utilities for data processing and analysis.
    """
    
    def __init__(self, autism_mode: bool = False, ptsd_mode: bool = False, 
                 combat_ptsd: bool = False, log_level: str = "INFO"):
        """
        Initialize the NeuroSim Engine
        
        Args:
            autism_mode: Enable autism-specific neural patterns
            ptsd_mode: Enable PTSD overlay mechanisms
            combat_ptsd: Enable combat-specific PTSD patterns
            log_level: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        self.autism_mode = autism_mode
        self.ptsd_mode = ptsd_mode
        self.combat_ptsd = combat_ptsd
        self.log_level = log_level
        
        if NEUROSIM_AVAILABLE:
            # Create configuration
            if combat_ptsd:
                self.config = neurosim_py.create_combat_ptsd_config()
                self.config.autism_mode = autism_mode
            elif autism_mode and ptsd_mode:
                self.config = neurosim_py.create_combined_config()
            elif autism_mode:
                self.config = neurosim_py.create_autism_config()
            elif ptsd_mode:
                self.config = neurosim_py.create_ptsd_config()
            else:
                self.config = neurosim_py.SimulatorConfig()
            
            self.config.log_level = log_level
            
            # Initialize simulator
            self.simulator = neurosim_py.NeuroSimulator(self.config)
            
            # Add combat trauma if specified
            if combat_ptsd:
                neurosim_py.add_fallujah_trauma_template(self.simulator)
                
        else:
            # Fallback simulation mode
            self.simulator = None
            self.config = None
            self._simulation_state = {
                "response_text": "Simulation mode",
                "region_activations": {},
                "timestamp": 0.0,
                "flashback_triggered": False
            }
    
    def process_text(self, text: str) -> Dict:
        """
        Process text input and return neural simulation results
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing simulation results
        """
        if NEUROSIM_AVAILABLE and self.simulator:
            result = self.simulator.process_text(text)
            return self._convert_simulation_state(result)
        else:
            # Fallback simulation
            return self._simulate_text_processing(text)
    
    def process_multimodal(self, text: str, visual_data: Optional[np.ndarray] = None,
                          audio_data: Optional[np.ndarray] = None,
                          motion_data: Optional[Dict] = None,
                          body_state: Optional[Dict] = None) -> Dict:
        """
        Process multi-modal input
        
        Args:
            text: Text input
            visual_data: Visual embedding or image data
            audio_data: Audio embedding or audio samples
            motion_data: Motion/vestibular data
            body_state: Interoceptive/body state data
            
        Returns:
            Dictionary containing simulation results
        """
        if NEUROSIM_AVAILABLE and self.simulator:
            # Create multi-modal input
            mm_input = neurosim_py.MultiModalInput()
            mm_input.text_tokens = text
            
            if visual_data is not None:
                mm_input.visual_embedding = visual_data
            else:
                mm_input.visual_embedding = np.zeros(512)
                
            if audio_data is not None:
                mm_input.audio_embedding = audio_data
            else:
                mm_input.audio_embedding = np.zeros(256)
                
            if motion_data is not None:
                mm_input.vestibular_embedding = np.array(motion_data.get('embedding', np.zeros(128)))
            else:
                mm_input.vestibular_embedding = np.zeros(128)
                
            if body_state is not None:
                mm_input.interoceptive_embedding = np.array(body_state.get('embedding', np.zeros(64)))
            else:
                mm_input.interoceptive_embedding = np.zeros(64)
            
            result = self.simulator.process(mm_input)
            return self._convert_simulation_state(result)
        else:
            # Fallback simulation
            return self._simulate_multimodal_processing(text, visual_data, audio_data, motion_data, body_state)
    
    def add_trauma_memory(self, trauma_description: str, sensitivity: float = 0.8):
        """
        Add a trauma memory for PTSD simulation
        
        Args:
            trauma_description: Description of traumatic experience
            sensitivity: Trigger sensitivity (0-1)
        """
        if NEUROSIM_AVAILABLE and self.simulator:
            # Create trauma embedding (simplified)
            trauma_embedding = np.random.randn(512) * 0.5
            self.simulator.add_trauma_memory(trauma_embedding, sensitivity)
        else:
            logging.info(f"Added trauma memory (simulation): {trauma_description}")
    
    def get_memory_traces(self) -> List[Dict]:
        """
        Get stored memory traces
        
        Returns:
            List of memory trace dictionaries
        """
        if NEUROSIM_AVAILABLE and self.simulator:
            traces = self.simulator.get_memory_traces()
            return [self._convert_simulation_state(trace) for trace in traces]
        else:
            return []
    
    def export_session_data(self, filename: str):
        """
        Export session data to JSON file
        
        Args:
            filename: Output filename
        """
        memory_traces = self.get_memory_traces()
        session_data = {
            "config": {
                "autism_mode": self.autism_mode,
                "ptsd_mode": self.ptsd_mode,
                "combat_ptsd": self.combat_ptsd,
                "log_level": self.log_level
            },
            "memory_traces": memory_traces,
            "total_interactions": len(memory_traces)
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def analyze_session(self) -> Dict:
        """
        Analyze current session data
        
        Returns:
            Dictionary containing session analysis
        """
        memory_traces = self.get_memory_traces()
        
        if not memory_traces:
            return {"error": "No memory traces available"}
        
        # Calculate statistics
        total_interactions = len(memory_traces)
        flashback_count = sum(1 for trace in memory_traces if trace.get('flashback_triggered', False))
        
        # Region activation analysis
        region_activations = {}
        for trace in memory_traces:
            for region, activation in trace.get('region_activations', {}).items():
                if region not in region_activations:
                    region_activations[region] = []
                region_activations[region].append(activation)
        
        avg_activations = {region: np.mean(activations) 
                          for region, activations in region_activations.items()}
        
        # Response analysis
        responses = [trace.get('response_text', '') for trace in memory_traces]
        unique_responses = list(set(responses))
        
        return {
            "total_interactions": total_interactions,
            "flashback_episodes": flashback_count,
            "flashback_rate": flashback_count / total_interactions if total_interactions > 0 else 0,
            "average_region_activations": avg_activations,
            "unique_responses": unique_responses,
            "most_common_response": max(set(responses), key=responses.count) if responses else None
        }
    
    def _convert_simulation_state(self, state) -> Dict:
        """Convert C++ simulation state to Python dictionary"""
        if NEUROSIM_AVAILABLE:
            return {
                "response_text": state.response_text,
                "region_activations": dict(state.region_activations),
                "timestamp": state.timestamp,
                "flashback_triggered": state.flashback_triggered,
                "active_memories": list(state.active_memories)
            }
        else:
            return state
    
    def _simulate_text_processing(self, text: str) -> Dict:
        """Fallback text processing simulation"""
        # Simple rule-based simulation
        response = "Okay."
        flashback = False
        
        # Check for trigger words
        trigger_words = ["loud", "explosion", "gun", "combat", "war", "danger", "threat"]
        if any(word in text.lower() for word in trigger_words):
            if self.ptsd_mode:
                response = "No. No. I don't want it."
                flashback = True
            else:
                response = "I'm scared."
        
        # Check for overwhelming stimuli
        overwhelming_words = ["too many", "crowd", "noise", "bright"]
        if any(phrase in text.lower() for phrase in overwhelming_words):
            if self.autism_mode:
                response = "Too much. Too much."
        
        # Simulate region activations
        region_activations = {
            "Amygdala": 0.8 if flashback else 0.3,
            "Hippocampus": 0.5,
            "Insula": 0.6 if self.autism_mode else 0.4,
            "PFC": 0.3 if flashback else 0.7,
            "STG": 0.4,
            "ACC": 0.6 if flashback else 0.4
        }
        
        self._simulation_state["timestamp"] += 1.0
        
        return {
            "response_text": response,
            "region_activations": region_activations,
            "timestamp": self._simulation_state["timestamp"],
            "flashback_triggered": flashback,
            "active_memories": ["trauma_memory_1"] if flashback else []
        }
    
    def _simulate_multimodal_processing(self, text: str, visual_data, audio_data, 
                                      motion_data, body_state) -> Dict:
        """Fallback multimodal processing simulation"""
        base_result = self._simulate_text_processing(text)
        
        # Modify based on additional modalities
        if visual_data is not None:
            base_result["region_activations"]["Amygdala"] += 0.1
        
        if audio_data is not None:
            base_result["region_activations"]["STG"] += 0.2
            if self.autism_mode:
                base_result["region_activations"]["Insula"] += 0.3
        
        if motion_data is not None:
            base_result["region_activations"]["Cerebellum"] = 0.6
        
        if body_state is not None:
            base_result["region_activations"]["Insula"] += 0.2
        
        return base_result


def create_autism_simulation() -> NeuroSimEngine:
    """Create a NeuroSim Engine configured for autism simulation"""
    return NeuroSimEngine(autism_mode=True, log_level="INFO")


def create_ptsd_simulation() -> NeuroSimEngine:
    """Create a NeuroSim Engine configured for PTSD simulation"""
    return NeuroSimEngine(ptsd_mode=True, log_level="INFO")


def create_combat_ptsd_simulation() -> NeuroSimEngine:
    """Create a NeuroSim Engine configured for combat PTSD simulation"""
    return NeuroSimEngine(ptsd_mode=True, combat_ptsd=True, log_level="INFO")


def create_combined_simulation() -> NeuroSimEngine:
    """Create a NeuroSim Engine configured for combined autism + PTSD simulation"""
    return NeuroSimEngine(autism_mode=True, ptsd_mode=True, log_level="INFO")


if __name__ == "__main__":
    # Example usage
    print("NeuroSim Engine Python Interface")
    print("=================================")
    
    # Test different configurations
    configs = [
        ("Normal", NeuroSimEngine()),
        ("Autism", create_autism_simulation()),
        ("PTSD", create_ptsd_simulation()),
        ("Combat PTSD", create_combat_ptsd_simulation()),
        ("Combined", create_combined_simulation())
    ]
    
    test_inputs = [
        "Hello, how are you?",
        "There are too many people here",
        "I heard a loud explosion",
        "Unknown person approaching"
    ]
    
    for config_name, engine in configs:
        print(f"\n{config_name} Configuration:")
        print("-" * 30)
        
        for text_input in test_inputs:
            result = engine.process_text(text_input)
            print(f"Input: '{text_input}'")
            print(f"Response: '{result['response_text']}'")
            print(f"Flashback: {result['flashback_triggered']}")
            print()
        
        # Show session analysis
        analysis = engine.analyze_session()
        print(f"Session Analysis: {analysis}")
        print()
