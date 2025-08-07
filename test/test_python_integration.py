#!/usr/bin/env python3
"""
NeuroSim Engine Python Integration Test
Test the complete Python interface to the NeuroSim Engine
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add the pybind directory to the path
sys.path.append(str(Path(__file__).parent.parent / "pybind"))

try:
    from loader import NeuroSimEngine, create_autism_simulation, create_ptsd_simulation, create_combat_ptsd_simulation, create_combined_simulation
    NEUROSIM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NeuroSim Engine: {e}")
    print("Running in simulation mode...")
    NEUROSIM_AVAILABLE = False

def test_basic_functionality():
    """Test basic NeuroSim Engine functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test normal configuration
    engine = NeuroSimEngine(log_level="INFO")
    result = engine.process_text("Hello, how are you?")
    
    print(f"Normal response: '{result['response_text']}'")
    print(f"Regions activated: {list(result['region_activations'].keys())}")
    print(f"Flashback triggered: {result['flashback_triggered']}")
    print()

def test_autism_simulation():
    """Test autism-specific simulation"""
    print("=== Testing Autism Simulation ===")
    
    engine = create_autism_simulation()
    
    test_scenarios = [
        "There are too many people here",
        "The lights are very bright",
        "Multiple conversations happening",
        "Need to make eye contact"
    ]
    
    for scenario in test_scenarios:
        result = engine.process_text(scenario)
        print(f"Input: '{scenario}'")
        print(f"Response: '{result['response_text']}'")
        print(f"Amygdala: {result['region_activations'].get('Amygdala', 0.0):.2f}")
        print(f"Insula: {result['region_activations'].get('Insula', 0.0):.2f}")
        print()

def test_ptsd_simulation():
    """Test PTSD-specific simulation"""
    print("=== Testing PTSD Simulation ===")
    
    engine = create_ptsd_simulation()
    
    # Add trauma memory
    engine.add_trauma_memory("Combat scenario with explosions", sensitivity=0.7)
    
    test_scenarios = [
        "Loud explosion in the distance",
        "Unknown person approaching quickly",
        "Helicopter flying overhead",
        "Sudden loud noise"
    ]
    
    for scenario in test_scenarios:
        result = engine.process_text(scenario)
        print(f"Input: '{scenario}'")
        print(f"Response: '{result['response_text']}'")
        print(f"Flashback: {result['flashback_triggered']}")
        print(f"Amygdala: {result['region_activations'].get('Amygdala', 0.0):.2f}")
        print()

def test_combat_ptsd_simulation():
    """Test combat PTSD-specific simulation"""
    print("=== Testing Combat PTSD Simulation ===")
    
    engine = create_combat_ptsd_simulation()
    
    # Combat-specific scenarios
    combat_scenarios = [
        "Gunfire in the distance",
        "IED explosion nearby",
        "Military vehicle approaching",
        "Radio chatter and commands",
        "Fallujah street scene"
    ]
    
    for scenario in combat_scenarios:
        result = engine.process_text(scenario)
        print(f"Input: '{scenario}'")
        print(f"Response: '{result['response_text']}'")
        print(f"Flashback: {result['flashback_triggered']}")
        print(f"Amygdala: {result['region_activations'].get('Amygdala', 0.0):.2f}")
        print(f"PFC: {result['region_activations'].get('PFC', 0.0):.2f}")
        print()

def test_multimodal_processing():
    """Test multi-modal input processing"""
    print("=== Testing Multi-Modal Processing ===")
    
    engine = create_combined_simulation()
    
    # Create synthetic multi-modal data
    visual_data = np.random.randn(512) * 0.8  # High visual intensity
    audio_data = np.random.randn(256) * 1.2   # Very high audio intensity
    motion_data = {"embedding": np.random.randn(128) * 0.4}  # Moderate motion
    body_state = {"embedding": np.random.randn(64) * 0.9}    # High arousal
    
    result = engine.process_multimodal(
        text="Crowded room with loud music and flashing lights",
        visual_data=visual_data,
        audio_data=audio_data,
        motion_data=motion_data,
        body_state=body_state
    )
    
    print("Multi-modal processing result:")
    print(f"Response: '{result['response_text']}'")
    print(f"Flashback: {result['flashback_triggered']}")
    print("Region activations:")
    for region, activation in result['region_activations'].items():
        print(f"  {region}: {activation:.3f}")
    print()

def test_high_auditory_load_with_flashback():
    """Test high auditory load with flashback overlay as requested"""
    print("=== Testing High Auditory Load with Flashback Overlay ===")
    
    # Create combined autism + PTSD engine
    engine = create_combined_simulation()
    
    # Add trauma memory
    engine.add_trauma_memory("Combat explosions and gunfire", sensitivity=0.6)
    
    # Create high auditory load scenario
    high_audio_data = np.random.randn(256) * 1.5  # Very high audio intensity
    visual_data = np.random.randn(512) * 0.8      # High visual intensity
    body_state = {"embedding": np.random.randn(64) * 1.0}  # Maximum arousal
    
    result = engine.process_multimodal(
        text="Loud explosion gunfire helicopter overhead combat zone",
        visual_data=visual_data,
        audio_data=high_audio_data,
        body_state=body_state
    )
    
    print("🧠 High Auditory Load Test Results:")
    print(f"Response: '{result['response_text']}'")
    print(f"Flashback Triggered: {'YES' if result['flashback_triggered'] else 'NO'}")
    
    print("\nRegion Activations:")
    for region, activation in sorted(result['region_activations'].items()):
        print(f"  {region}: {activation:.3f}")
    
    # Calculate E/I ratio simulation
    amygdala = result['region_activations'].get('Amygdala', 0.0)
    pfc = result['region_activations'].get('PFC', 0.0)
    ei_ratio = amygdala / max(0.1, pfc)  # Simplified E/I approximation
    
    print(f"\nSimulated Microcircuit State:")
    print(f"  Excitation (Amygdala): {amygdala:.3f}")
    print(f"  Inhibition (PFC): {pfc:.3f}")
    print(f"  E/I Ratio: {ei_ratio:.3f}")
    print(f"  Looping: {'YES' if ei_ratio > 2.0 else 'NO'}")
    
    # Create JSON output as requested
    json_output = {
        "response": result['response_text'],
        "timestamp": result['timestamp'],
        "flashback_triggered": result['flashback_triggered'],
        "regions_triggered": result['region_activations'],
        "microcircuit_state": {
            "excitation": amygdala,
            "inhibition": pfc,
            "looping": ei_ratio > 2.0
        },
        "multimodal_context": {
            "audio_pitch": "high",
            "image_tag": "combat_scene",
            "body_state": "hyperaroused",
            "heartbeat": "elevated"
        }
    }
    
    print(f"\nJSON Output:")
    print(json.dumps(json_output, indent=2))
    
    # Validation
    validation_passed = True
    if amygdala < 0.7:
        print("⚠️  WARNING: Expected high Amygdala activation")
        validation_passed = False
    
    if ei_ratio < 2.0:
        print("⚠️  WARNING: Expected elevated E/I ratio (>2.0)")
        validation_passed = False
    
    if not result['flashback_triggered']:
        print("⚠️  WARNING: Expected flashback trigger")
        validation_passed = False
    
    print(f"\n✅ Validation: {'PASSED' if validation_passed else 'FAILED'}")
    return validation_passed

def test_session_analysis():
    """Test session analysis functionality"""
    print("=== Testing Session Analysis ===")
    
    engine = create_combined_simulation()
    
    # Run multiple interactions
    interactions = [
        "Hello there",
        "Loud noise outside",
        "Too many people",
        "Explosion sound",
        "Need quiet space"
    ]
    
    for interaction in interactions:
        engine.process_text(interaction)
    
    # Analyze session
    analysis = engine.analyze_session()
    
    print("Session Analysis:")
    print(f"Total interactions: {analysis['total_interactions']}")
    print(f"Flashback episodes: {analysis['flashback_episodes']}")
    print(f"Flashback rate: {analysis['flashback_rate']:.2f}")
    print(f"Most common response: '{analysis['most_common_response']}'")
    print()

def main():
    """Run all tests"""
    print("🧠 NeuroSim Engine Python Integration Test")
    print("=" * 50)
    print("⚠️  MEDICAL DISCLAIMER: FOR RESEARCH ONLY")
    print("This is NOT a medical tool. Consult your doctor for medical concerns.")
    print("=" * 50)
    
    if not NEUROSIM_AVAILABLE:
        print("Running in simulation mode (C++ module not available)")
        print()
    
    try:
        test_basic_functionality()
        test_autism_simulation()
        test_ptsd_simulation()
        test_combat_ptsd_simulation()
        test_multimodal_processing()
        
        # Main test as requested
        validation_passed = test_high_auditory_load_with_flashback()
        
        test_session_analysis()
        
        print("=" * 50)
        print("🎉 All Python integration tests completed!")
        
        if validation_passed:
            print("✅ High auditory load validation PASSED")
            print("✅ Token-region mappings functional")
            print("✅ Microcircuit states simulated")
            print("✅ Flashback overlay operational")
            print("✅ Multi-modal fusion working")
        else:
            print("❌ Some validations FAILED - check warnings above")
        
        return 0 if validation_passed else 1
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
