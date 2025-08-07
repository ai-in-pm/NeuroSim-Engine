#include "microcircuit.hpp"
#include <algorithm>
#include <random>
#include <cmath>

namespace neurosim {

MicroCircuit::MicroCircuit(const CircuitConfig& config) 
    : config_(config), current_time_(0.0), flashback_start_time_(0.0) {
    
    // Initialize baseline state
    current_state_.excitatory_activity = config_.baseline_excitation;
    current_state_.inhibitory_activity = config_.baseline_inhibition;
    current_state_.neurotransmitters.glutamate_level = 1.0;
    current_state_.neurotransmitters.gaba_level = 1.0;
    
    // Apply autism modifications if enabled
    if (config_.autism_mode) {
        enableAutismMode();
    }
    
    // Apply PTSD modifications if enabled
    if (config_.ptsd_mode) {
        enablePTSDMode();
    }
}

MicroCircuit::ActivationState MicroCircuit::process(double input_strength, double dt) {
    current_time_ += dt;
    
    // Update excitatory activity
    updateExcitatoryActivity(input_strength, dt);
    
    // Update inhibitory activity (with potential delay)
    updateInhibitoryActivity(dt);
    
    // Update neurotransmitter levels
    updateNeurotransmitters(dt);
    
    // Calculate net activation
    current_state_.net_activation = current_state_.excitatory_activity - current_state_.inhibitory_activity;
    
    // Calculate firing rate
    current_state_.firing_rate = calculateFiringRate(current_state_.net_activation);
    
    // Apply adaptation
    applyAdaptation(dt);
    
    // Add neural noise
    addNoise(dt);
    
    // Apply autism-specific modifications
    if (config_.autism_mode) {
        applyAutismModifications();
    }
    
    // Apply PTSD-specific modifications
    if (config_.ptsd_mode) {
        applyPTSDModifications();
    }
    
    // Detect oscillations and pathological patterns
    detectOscillations();
    current_state_.hyperexcitable = detectHyperexcitability();
    current_state_.inhibition_failure = detectInhibitionFailure();
    
    // Update activation history
    updateActivationHistory();
    
    return current_state_;
}

void MicroCircuit::updateExcitatoryActivity(double input_strength, double dt) {
    // Simple excitatory dynamics with glutamate modulation
    double target_excitation = config_.baseline_excitation + 
                              input_strength * current_state_.neurotransmitters.glutamate_level;
    
    // Apply E/I ratio modification
    target_excitation *= config_.ei_ratio;
    
    // Exponential approach to target
    double tau_excitation = 10.0; // ms
    current_state_.excitatory_activity += 
        (target_excitation - current_state_.excitatory_activity) * dt / tau_excitation;
    
    // Apply bounds
    current_state_.excitatory_activity = std::max(0.0, 
        std::min(5.0, current_state_.excitatory_activity));
}

void MicroCircuit::updateInhibitoryActivity(double dt) {
    // Inhibitory activity follows excitatory with delay
    double target_inhibition = current_state_.excitatory_activity * 
                              current_state_.neurotransmitters.gaba_level;
    
    // Apply inhibition delay (increased in PTSD)
    double effective_delay = config_.inhibition_delay_ms;
    if (config_.ptsd_mode) {
        effective_delay = config_.ptsd_inhibition_delay;
    }
    
    // Simple delay model: slower response to excitation
    double tau_inhibition = 20.0 + effective_delay; // ms
    current_state_.inhibitory_activity += 
        (target_inhibition - current_state_.inhibitory_activity) * dt / tau_inhibition;
    
    // Apply autism inhibition deficit
    if (config_.autism_mode) {
        current_state_.inhibitory_activity *= config_.autism_inhibition_deficit;
    }
    
    // Apply bounds
    current_state_.inhibitory_activity = std::max(0.0, 
        std::min(3.0, current_state_.inhibitory_activity));
}

void MicroCircuit::updateNeurotransmitters(double dt) {
    // Simple neurotransmitter dynamics
    double tau_nt = 100.0; // ms
    
    // Glutamate increases with excitatory activity
    double target_glutamate = 1.0 + current_state_.excitatory_activity * 0.2;
    current_state_.neurotransmitters.glutamate_level += 
        (target_glutamate - current_state_.neurotransmitters.glutamate_level) * dt / tau_nt;
    
    // GABA increases with inhibitory activity
    double target_gaba = 1.0 + current_state_.inhibitory_activity * 0.15;
    current_state_.neurotransmitters.gaba_level += 
        (target_gaba - current_state_.neurotransmitters.gaba_level) * dt / tau_nt;
    
    // Apply bounds
    current_state_.neurotransmitters.glutamate_level = 
        std::max(0.1, std::min(2.0, current_state_.neurotransmitters.glutamate_level));
    current_state_.neurotransmitters.gaba_level = 
        std::max(0.1, std::min(2.0, current_state_.neurotransmitters.gaba_level));
}

void MicroCircuit::applyAdaptation(double dt) {
    // Neural adaptation reduces response over time
    double adaptation_target = current_state_.firing_rate * 0.1;
    double tau_adaptation = 500.0; // ms
    
    current_state_.adaptation_level += 
        (adaptation_target - current_state_.adaptation_level) * dt / tau_adaptation;
    
    // Apply adaptation to excitatory activity
    current_state_.excitatory_activity *= (1.0 - current_state_.adaptation_level * config_.adaptation_rate);
}

void MicroCircuit::addNoise(double dt) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> noise_dist(0.0, 1.0);
    
    double noise_strength = config_.noise_level * std::sqrt(dt);
    current_state_.excitatory_activity += noise_dist(gen) * noise_strength;
    current_state_.inhibitory_activity += noise_dist(gen) * noise_strength * 0.5;
    
    // Ensure non-negative values
    current_state_.excitatory_activity = std::max(0.0, current_state_.excitatory_activity);
    current_state_.inhibitory_activity = std::max(0.0, current_state_.inhibitory_activity);
}

void MicroCircuit::applyAutismModifications() {
    // Enhanced E/I ratio
    current_state_.excitatory_activity *= config_.autism_ei_elevation;
    
    // Reduced inhibitory control
    current_state_.inhibitory_activity *= config_.autism_inhibition_deficit;
}

void MicroCircuit::applyPTSDModifications() {
    // Hyperarousal
    current_state_.excitatory_activity *= config_.ptsd_hyperarousal;
    
    // Check for memory intrusion
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> intrusion_dist(0.0, 1.0);
    
    if (intrusion_dist(gen) < config_.ptsd_memory_intrusion) {
        // Simulate memory intrusion as sudden excitatory burst
        current_state_.excitatory_activity += 1.0;
    }
}

void MicroCircuit::detectOscillations() {
    if (current_state_.activation_history.size() < 10) {
        current_state_.in_oscillation = false;
        return;
    }
    
    // Simple oscillation detection: look for regular peaks
    std::vector<double> recent_history(current_state_.activation_history.end() - 10, 
                                     current_state_.activation_history.end());
    
    // Count zero crossings as a simple oscillation measure
    int zero_crossings = 0;
    double mean = std::accumulate(recent_history.begin(), recent_history.end(), 0.0) / recent_history.size();
    
    for (size_t i = 1; i < recent_history.size(); ++i) {
        if ((recent_history[i-1] - mean) * (recent_history[i] - mean) < 0) {
            zero_crossings++;
        }
    }
    
    current_state_.in_oscillation = zero_crossings > 4;
    if (current_state_.in_oscillation) {
        current_state_.oscillation_frequency = calculateOscillationFrequency();
    }
}

double MicroCircuit::calculateOscillationFrequency() const {
    // Simplified frequency calculation
    if (current_state_.activation_history.size() < 20) return 0.0;
    
    // Estimate frequency from zero crossings in recent history
    std::vector<double> recent_history(current_state_.activation_history.end() - 20, 
                                     current_state_.activation_history.end());
    
    double mean = std::accumulate(recent_history.begin(), recent_history.end(), 0.0) / recent_history.size();
    int zero_crossings = 0;
    
    for (size_t i = 1; i < recent_history.size(); ++i) {
        if ((recent_history[i-1] - mean) * (recent_history[i] - mean) < 0) {
            zero_crossings++;
        }
    }
    
    // Frequency = zero_crossings / (2 * time_window)
    double time_window = 20.0; // ms (assuming 1ms per sample)
    return (zero_crossings / 2.0) * (1000.0 / time_window); // Convert to Hz
}

bool MicroCircuit::detectHyperexcitability() const {
    return current_state_.excitatory_activity > 3.0 || 
           (current_state_.excitatory_activity / std::max(0.1, current_state_.inhibitory_activity)) > 3.0;
}

bool MicroCircuit::detectInhibitionFailure() const {
    return current_state_.inhibitory_activity < 0.2 && current_state_.excitatory_activity > 1.0;
}

double MicroCircuit::calculateFiringRate(double net_activation) const {
    // Sigmoid activation function
    double sigmoid_output = 1.0 / (1.0 + std::exp(-net_activation));
    return sigmoid_output * MAX_FIRING_RATE;
}

void MicroCircuit::updateActivationHistory() {
    current_state_.activation_history.push_back(current_state_.net_activation);
    
    // Limit history size
    if (current_state_.activation_history.size() > MAX_HISTORY_SIZE) {
        current_state_.activation_history.erase(current_state_.activation_history.begin());
    }
    
    // Store in main history
    activation_history_.push_back(current_state_);
    if (activation_history_.size() > MAX_HISTORY_SIZE) {
        activation_history_.erase(activation_history_.begin());
    }
}

void MicroCircuit::enableAutismMode() {
    config_.autism_mode = true;
    config_.ei_ratio = config_.autism_ei_elevation;
    config_.baseline_inhibition *= config_.autism_inhibition_deficit;
}

void MicroCircuit::enablePTSDMode() {
    config_.ptsd_mode = true;
    config_.inhibition_delay_ms = config_.ptsd_inhibition_delay;
    config_.baseline_excitation *= config_.ptsd_hyperarousal;
}

void MicroCircuit::reset() {
    current_state_ = ActivationState{};
    current_state_.excitatory_activity = config_.baseline_excitation;
    current_state_.inhibitory_activity = config_.baseline_inhibition;
    current_state_.neurotransmitters.glutamate_level = 1.0;
    current_state_.neurotransmitters.gaba_level = 1.0;
    
    activation_history_.clear();
    current_time_ = 0.0;
}

std::vector<MicroCircuit::ActivationState> MicroCircuit::getActivationHistory() const {
    return activation_history_;
}

void MicroCircuit::updateConfig(const CircuitConfig& config) {
    config_ = config;
}

// BrainRegion implementation
BrainRegion::BrainRegion(const RegionConfig& config) 
    : config_(config), current_activation_(0.0), current_time_(0.0) {
    
    microcircuit_ = std::make_unique<MicroCircuit>(config.circuit_config);
}

const MicroCircuit::ActivationState& BrainRegion::getMicrocircuitState() const {
    return microcircuit_->getCurrentState();
}

} // namespace neurosim
