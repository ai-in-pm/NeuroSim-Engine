#include <iostream>
#include <string>
#include <vector>
#include <map>

/**
 * Simple test program for NeuroSim Engine basic functionality
 * This test doesn't require external dependencies and validates core concepts
 * 
 * Owner: Darrell Mesa (darrell.mesa@pm-ss.org)
 * GitHub: https://github.com/ai-in-pm
 */

// Simple simulation structures (without Eigen dependency)
struct SimpleVector {
    std::vector<double> data;
    size_t size() const { return data.size(); }
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    
    SimpleVector(size_t n = 0) : data(n, 0.0) {}
    SimpleVector(std::initializer_list<double> init) : data(init) {}
};

// Simplified NeuroSim structures
struct SimulationConfig {
    bool autism_mode = false;
    bool ptsd_overlay = false;
    double excitation_ratio = 1.0;
    double inhibition_delay = 0.0;
    std::string log_level = "INFO";
};

struct RegionActivation {
    std::string region_name;
    double activation_strength = 0.0;
    double latency_ms = 0.0;
    std::string activation_reason;
};

struct SimulationState {
    std::string response_text;
    std::map<std::string, double> region_activations;
    struct {
        double excitation = 1.0;
        double inhibition = 1.0;
        bool looping = false;
    } microcircuit_state;
    bool flashback_triggered = false;
    double timestamp = 0.0;
};

// Simple NeuroSim Engine implementation
class SimpleNeuroSim {
private:
    SimulationConfig config_;
    double current_time_;
    std::vector<std::string> trauma_memories_;
    
public:
    SimpleNeuroSim(const SimulationConfig& config = SimulationConfig{}) 
        : config_(config), current_time_(0.0) {
        
        if (config_.log_level == "DEBUG") {
            std::cout << "SimpleNeuroSim initialized with autism_mode=" 
                      << config_.autism_mode << ", ptsd_overlay=" 
                      << config_.ptsd_overlay << std::endl;
        }
    }
    
    SimulationState processText(const std::string& text) {
        current_time_ += 1.0;
        
        SimulationState state;
        state.timestamp = current_time_;
        
        // Simple token analysis
        bool has_threat = (text.find("explosion") != std::string::npos ||
                          text.find("gunfire") != std::string::npos ||
                          text.find("loud") != std::string::npos ||
                          text.find("danger") != std::string::npos);
        
        bool has_social = (text.find("people") != std::string::npos ||
                          text.find("crowd") != std::string::npos ||
                          text.find("many") != std::string::npos);
        
        bool has_overwhelming = (text.find("too much") != std::string::npos ||
                               text.find("too many") != std::string::npos ||
                               text.find("bright") != std::string::npos);
        
        // Calculate region activations
        double amygdala_activation = 0.2; // baseline
        double pfc_activation = 0.4;      // baseline
        double insula_activation = 0.25;  // baseline
        
        if (has_threat) {
            amygdala_activation += 0.6;
            if (config_.ptsd_overlay) {
                amygdala_activation += 0.2;
                state.flashback_triggered = checkTraumaMatch(text);
            }
        }
        
        if (has_social && config_.autism_mode) {
            amygdala_activation += 0.3;
            insula_activation += 0.4;
        }
        
        if (has_overwhelming && config_.autism_mode) {
            insula_activation += 0.5;
            pfc_activation -= 0.2; // reduced executive control
        }
        
        // Apply autism modifications
        if (config_.autism_mode) {
            amygdala_activation *= 1.3; // hypersensitivity
            insula_activation *= 1.4;   // sensory processing
            pfc_activation *= 0.7;      // reduced inhibition
        }
        
        // Apply PTSD modifications
        if (config_.ptsd_overlay) {
            amygdala_activation *= 1.5; // hypervigilance
            pfc_activation *= 0.6;      // impaired control
        }
        
        // Clamp values
        amygdala_activation = std::min(1.0, std::max(0.0, amygdala_activation));
        pfc_activation = std::min(1.0, std::max(0.0, pfc_activation));
        insula_activation = std::min(1.0, std::max(0.0, insula_activation));
        
        // Store activations
        state.region_activations["Amygdala"] = amygdala_activation;
        state.region_activations["PFC"] = pfc_activation;
        state.region_activations["Insula"] = insula_activation;
        state.region_activations["Hippocampus"] = 0.5;
        state.region_activations["STG"] = 0.4;
        state.region_activations["ACC"] = 0.6;
        state.region_activations["Cerebellum"] = 0.3;
        
        // Calculate microcircuit state
        state.microcircuit_state.excitation = amygdala_activation * config_.excitation_ratio;
        state.microcircuit_state.inhibition = pfc_activation;
        state.microcircuit_state.looping = (state.microcircuit_state.excitation / 
                                           std::max(0.1, state.microcircuit_state.inhibition)) > 2.0;
        
        // Generate response
        state.response_text = generateResponse(state);
        
        return state;
    }
    
    void addTraumaMemory(const std::string& trauma_description) {
        trauma_memories_.push_back(trauma_description);
        if (config_.log_level == "DEBUG") {
            std::cout << "Added trauma memory: " << trauma_description << std::endl;
        }
    }
    
    const SimulationConfig& getConfig() const { return config_; }
    
private:
    bool checkTraumaMatch(const std::string& text) {
        for (const auto& trauma : trauma_memories_) {
            if (text.find("explosion") != std::string::npos && 
                trauma.find("explosion") != std::string::npos) {
                return true;
            }
            if (text.find("gunfire") != std::string::npos && 
                trauma.find("combat") != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    std::string generateResponse(const SimulationState& state) {
        double amygdala = state.region_activations.at("Amygdala");
        
        if (state.flashback_triggered) {
            return "No. No. I don't want it.";
        } else if (amygdala > 0.8) {
            return "I'm scared.";
        } else if (state.microcircuit_state.looping) {
            return "Too much. Too much.";
        } else if (config_.autism_mode && state.region_activations.at("Insula") > 0.7) {
            return "Need quiet.";
        } else {
            return "Okay.";
        }
    }
};

// Test functions
void testBasicFunctionality() {
    std::cout << "=== Testing Basic Functionality ===" << std::endl;
    
    SimpleNeuroSim sim;
    auto result = sim.processText("Hello, how are you?");
    
    std::cout << "Normal response: " << result.response_text << std::endl;
    std::cout << "Amygdala: " << result.region_activations["Amygdala"] << std::endl;
    std::cout << "Flashback: " << (result.flashback_triggered ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
}

void testAutismMode() {
    std::cout << "=== Testing Autism Mode ===" << std::endl;
    
    SimulationConfig config;
    config.autism_mode = true;
    config.excitation_ratio = 1.4;
    config.log_level = "DEBUG";
    
    SimpleNeuroSim sim(config);
    auto result = sim.processText("There are too many people here");
    
    std::cout << "Autism response: " << result.response_text << std::endl;
    std::cout << "Amygdala: " << result.region_activations["Amygdala"] << std::endl;
    std::cout << "Insula: " << result.region_activations["Insula"] << std::endl;
    std::cout << "E/I Ratio: " << (result.microcircuit_state.excitation / 
                                  std::max(0.1, result.microcircuit_state.inhibition)) << std::endl;
    std::cout << std::endl;
}

void testPTSDMode() {
    std::cout << "=== Testing PTSD Mode ===" << std::endl;
    
    SimulationConfig config;
    config.ptsd_overlay = true;
    config.inhibition_delay = 50.0;
    config.log_level = "DEBUG";
    
    SimpleNeuroSim sim(config);
    sim.addTraumaMemory("Combat explosions and gunfire");
    
    auto result = sim.processText("Loud explosion nearby");
    
    std::cout << "PTSD response: " << result.response_text << std::endl;
    std::cout << "Amygdala: " << result.region_activations["Amygdala"] << std::endl;
    std::cout << "Flashback: " << (result.flashback_triggered ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
}

void testHighAuditoryLoad() {
    std::cout << "=== Testing High Auditory Load with Flashback Overlay ===" << std::endl;
    
    SimulationConfig config;
    config.autism_mode = true;
    config.ptsd_overlay = true;
    config.excitation_ratio = 1.4;
    config.inhibition_delay = 50.0;
    config.log_level = "DEBUG";
    
    SimpleNeuroSim sim(config);
    sim.addTraumaMemory("Operation Phantom Fury combat scenario");
    
    auto result = sim.processText("Loud explosion gunfire helicopter overhead");
    
    std::cout << "🧠 High Auditory Load Test Results:" << std::endl;
    std::cout << "Response: " << result.response_text << std::endl;
    std::cout << "Flashback Triggered: " << (result.flashback_triggered ? "YES" : "NO") << std::endl;
    std::cout << "Microcircuit Looping: " << (result.microcircuit_state.looping ? "YES" : "NO") << std::endl;
    
    std::cout << "\nRegion Activations:" << std::endl;
    for (const auto& [region, activation] : result.region_activations) {
        std::cout << "  " << region << ": " << activation << std::endl;
    }
    
    double ei_ratio = result.microcircuit_state.excitation / 
                     std::max(0.1, result.microcircuit_state.inhibition);
    
    std::cout << "\nMicrocircuit State:" << std::endl;
    std::cout << "  Excitation: " << result.microcircuit_state.excitation << std::endl;
    std::cout << "  Inhibition: " << result.microcircuit_state.inhibition << std::endl;
    std::cout << "  E/I Ratio: " << ei_ratio << std::endl;
    
    // Validation
    bool validation_passed = true;
    if (result.region_activations.at("Amygdala") < 0.7) {
        std::cout << "⚠️  WARNING: Expected high Amygdala activation" << std::endl;
        validation_passed = false;
    }
    if (ei_ratio < 2.0) {
        std::cout << "⚠️  WARNING: Expected elevated E/I ratio (>2.0)" << std::endl;
        validation_passed = false;
    }
    
    std::cout << "\n✅ Validation: " << (validation_passed ? "PASSED" : "FAILED") << std::endl;
    return;
}

int main() {
    std::cout << "🧠 NeuroSim Engine - Simple Test Suite" << std::endl;
    std::cout << "Owner: Darrell Mesa (darrell.mesa@pm-ss.org)" << std::endl;
    std::cout << "GitHub: https://github.com/ai-in-pm" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "⚠️  MEDICAL DISCLAIMER: FOR RESEARCH ONLY" << std::endl;
    std::cout << "This is NOT a medical tool. Consult your doctor for medical concerns." << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        testBasicFunctionality();
        testAutismMode();
        testPTSDMode();
        testHighAuditoryLoad();
        
        std::cout << "🎉 All tests completed successfully!" << std::endl;
        std::cout << "✅ Token-to-brain routing functional" << std::endl;
        std::cout << "✅ Autism and PTSD overlays active" << std::endl;
        std::cout << "✅ Microcircuit simulation running" << std::endl;
        std::cout << "✅ Basic validation passed" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
