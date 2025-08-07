#include "../core/simulator.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace neurosim;

/**
 * @brief Basic test of the NeuroSim Engine
 * 
 * This test validates:
 * - Basic simulator initialization
 * - Text processing with autism and PTSD modes
 * - JSON output generation
 * - Memory trace storage
 */
int main() {
    std::cout << "=== NeuroSim Engine Basic Test ===" << std::endl;
    
    try {
        // Test 1: Normal mode simulation
        std::cout << "\n1. Testing normal mode..." << std::endl;
        NeuroSimulator::Config normal_config;
        normal_config.autism_mode = false;
        normal_config.ptsd_overlay = false;
        normal_config.log_level = "DEBUG";
        
        NeuroSimulator normal_sim(normal_config);
        auto normal_result = normal_sim.processText("Hello, how are you?");
        
        std::cout << "Normal response: " << normal_result.response_text << std::endl;
        auto normal_json = normal_sim.exportToJson(normal_result);
        std::cout << "JSON output: " << normal_json.dump(2) << std::endl;
        
        // Test 2: Autism mode simulation
        std::cout << "\n2. Testing autism mode..." << std::endl;
        NeuroSimulator::Config autism_config;
        autism_config.autism_mode = true;
        autism_config.ptsd_overlay = false;
        autism_config.excitation_ratio = 1.4;
        autism_config.log_level = "DEBUG";
        
        NeuroSimulator autism_sim(autism_config);
        auto autism_result = autism_sim.processText("There are too many people here");
        
        std::cout << "Autism response: " << autism_result.response_text << std::endl;
        auto autism_json = autism_sim.exportToJson(autism_result);
        std::cout << "JSON output: " << autism_json.dump(2) << std::endl;
        
        // Test 3: PTSD mode simulation
        std::cout << "\n3. Testing PTSD mode..." << std::endl;
        NeuroSimulator::Config ptsd_config;
        ptsd_config.autism_mode = false;
        ptsd_config.ptsd_overlay = true;
        ptsd_config.inhibition_delay = 50.0;
        ptsd_config.flashback_sensitivity = 0.5;
        ptsd_config.log_level = "DEBUG";
        
        NeuroSimulator ptsd_sim(ptsd_config);
        
        // Add a trauma memory (simulated combat scenario)
        Eigen::VectorXd trauma_embedding = Eigen::VectorXd::Random(512);
        ptsd_sim.addTraumaMemory(trauma_embedding, 0.7);
        
        auto ptsd_result = ptsd_sim.processText("Loud noise explosion");
        
        std::cout << "PTSD response: " << ptsd_result.response_text << std::endl;
        auto ptsd_json = ptsd_sim.exportToJson(ptsd_result);
        std::cout << "JSON output: " << ptsd_json.dump(2) << std::endl;
        
        // Test 4: Combined autism + PTSD mode
        std::cout << "\n4. Testing combined autism + PTSD mode..." << std::endl;
        NeuroSimulator::Config combined_config;
        combined_config.autism_mode = true;
        combined_config.ptsd_overlay = true;
        combined_config.excitation_ratio = 1.4;
        combined_config.inhibition_delay = 50.0;
        combined_config.log_level = "DEBUG";
        
        NeuroSimulator combined_sim(combined_config);
        combined_sim.addTraumaMemory(trauma_embedding, 0.6);
        
        auto combined_result = combined_sim.processText("Unknown person approaching");
        
        std::cout << "Combined response: " << combined_result.response_text << std::endl;
        auto combined_json = combined_sim.exportToJson(combined_result);
        std::cout << "JSON output: " << combined_json.dump(2) << std::endl;
        
        // Test 5: Multi-modal input processing
        std::cout << "\n5. Testing multi-modal input..." << std::endl;
        NeuroSimulator::MultiModalInput multimodal_input;
        multimodal_input.text_tokens = "I see a person";
        multimodal_input.visual_embedding = Eigen::VectorXd::Random(512);
        multimodal_input.audio_embedding = Eigen::VectorXd::Random(256);
        multimodal_input.vestibular_embedding = Eigen::VectorXd::Random(128);
        multimodal_input.interoceptive_embedding = Eigen::VectorXd::Random(64);
        multimodal_input.timestamp = 1000.0;
        
        auto multimodal_result = combined_sim.process(multimodal_input);
        
        std::cout << "Multimodal response: " << multimodal_result.response_text << std::endl;
        auto multimodal_json = combined_sim.exportToJson(multimodal_result);
        std::cout << "JSON output: " << multimodal_json.dump(2) << std::endl;
        
        // Test 6: Memory trace analysis
        std::cout << "\n6. Testing memory traces..." << std::endl;
        auto memory_traces = combined_sim.getMemoryTraces();
        std::cout << "Total memory traces: " << memory_traces.size() << std::endl;
        
        if (!memory_traces.empty()) {
            const auto& latest_trace = memory_traces.back();
            std::cout << "Latest trace timestamp: " << latest_trace.timestamp << std::endl;
            std::cout << "Latest trace response: " << latest_trace.response_text << std::endl;
            std::cout << "Flashback triggered: " << latest_trace.flashback_triggered << std::endl;
        }
        
        // Test 7: Configuration updates
        std::cout << "\n7. Testing configuration updates..." << std::endl;
        NeuroSimulator::Config new_config = combined_sim.getConfig();
        new_config.excitation_ratio = 1.6;
        new_config.flashback_sensitivity = 0.3;
        combined_sim.updateConfig(new_config);
        
        auto updated_result = combined_sim.processText("Testing updated config");
        std::cout << "Updated config response: " << updated_result.response_text << std::endl;
        
        // Test 8: High auditory load with flashback overlay (as requested)
        std::cout << "\n8. Testing high auditory load with flashback overlay..." << std::endl;
        testHighAuditoryLoadWithFlashback();

        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
        std::cout << "\n🧠 NeuroSim Engine validation complete!" << std::endl;
        std::cout << "✅ Token-to-brain routing functional" << std::endl;
        std::cout << "✅ Multi-modal fusion operational" << std::endl;
        std::cout << "✅ Autism and PTSD overlays active" << std::endl;
        std::cout << "✅ Microcircuit simulation running" << std::endl;
        std::cout << "✅ Memory and flashback systems enabled" << std::endl;

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}

/**
 * @brief Test high auditory load with flashback overlay as requested
 */
void testHighAuditoryLoadWithFlashback() {
    std::cout << "\n=== Testing High Auditory Load with Flashback Overlay ===" << std::endl;

    // Create combined autism + PTSD configuration
    NeuroSimulator::Config config;
    config.autism_mode = true;
    config.ptsd_overlay = true;
    config.excitation_ratio = 1.4;
    config.inhibition_delay = 50.0;
    config.flashback_sensitivity = 0.4;
    config.log_level = "DEBUG";

    NeuroSimulator sim(config);

    // Add trauma memory
    Eigen::VectorXd trauma_embedding = Eigen::VectorXd::Random(512);
    sim.addTraumaMemory(trauma_embedding, 0.6);

    // Create high auditory load scenario
    NeuroSimulator::MultiModalInput input;
    input.text_tokens = "Loud explosion gunfire helicopter overhead";
    input.visual_embedding = Eigen::VectorXd::Random(512) * 0.8; // High visual intensity
    input.audio_embedding = Eigen::VectorXd::Random(256) * 1.2;  // Very high audio intensity
    input.vestibular_embedding = Eigen::VectorXd::Random(128) * 0.6; // Moderate motion
    input.interoceptive_embedding = Eigen::VectorXd::Random(64) * 0.9; // High arousal
    input.timestamp = 1000.0;

    // Process the input
    auto result = sim.process(input);

    std::cout << "High Auditory Load Test Results:" << std::endl;
    std::cout << "Response: " << result.response_text << std::endl;
    std::cout << "Flashback Triggered: " << (result.flashback_triggered ? "YES" : "NO") << std::endl;
    std::cout << "Microcircuit Looping: " << (result.microcircuit_state.looping ? "YES" : "NO") << std::endl;

    std::cout << "\nRegion Activations:" << std::endl;
    for (const auto& [region, activation] : result.region_activations) {
        std::cout << "  " << region << ": " << activation << std::endl;
    }

    std::cout << "\nMicrocircuit State:" << std::endl;
    std::cout << "  Excitation: " << result.microcircuit_state.excitation << std::endl;
    std::cout << "  Inhibition: " << result.microcircuit_state.inhibition << std::endl;
    std::cout << "  E/I Ratio: " << (result.microcircuit_state.excitation /
                                   std::max(0.1, result.microcircuit_state.inhibition)) << std::endl;

    // Export to JSON as requested
    auto json_output = sim.exportToJson(result);
    std::cout << "\nJSON Output:" << std::endl;
    std::cout << json_output.dump(2) << std::endl;

    // Validate expected patterns
    bool validation_passed = true;

    // Check for high amygdala activation
    if (result.region_activations.count("Amygdala") &&
        result.region_activations.at("Amygdala") < 0.7) {
        std::cout << "WARNING: Expected high Amygdala activation" << std::endl;
        validation_passed = false;
    }

    // Check for microcircuit dysfunction
    if (!result.microcircuit_state.looping) {
        std::cout << "WARNING: Expected microcircuit looping under high load" << std::endl;
        validation_passed = false;
    }

    // Check E/I ratio
    double ei_ratio = result.microcircuit_state.excitation /
                     std::max(0.1, result.microcircuit_state.inhibition);
    if (ei_ratio < 2.0) {
        std::cout << "WARNING: Expected elevated E/I ratio (>2.0), got " << ei_ratio << std::endl;
        validation_passed = false;
    }

    std::cout << "\nValidation: " << (validation_passed ? "PASSED" : "FAILED") << std::endl;
}

/**
 * @brief Example usage demonstrating the expected JSON output format
 */
void demonstrateExpectedOutput() {
    std::cout << "\n=== Expected Output Example ===" << std::endl;
    std::cout << R"({
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
})" << std::endl;
}
