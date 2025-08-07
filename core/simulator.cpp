#include "simulator.hpp"
#include "brain_router.hpp"
#include "multimodal_fusion.hpp"
#include "memory_overlay.hpp"
#include "flashback_overlay.hpp"
#include "../regions/amygdala.hpp"
#include "../regions/hippocampus.hpp"
#include "../regions/insula.hpp"
#include "../regions/prefrontal.hpp"
#include "../regions/cerebellum.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

// Temporary implementations for missing classes
class MemoryOverlay {
public:
    MemoryOverlay() = default;
    void clearMemory() {}
};

class FlashbackOverlay {
public:
    FlashbackOverlay() = default;
    bool checkTrigger(const Eigen::VectorXd&) { return false; }
    void addTraumaTemplate(const Eigen::VectorXd&, double) {}
};

namespace neurosim {

NeuroSimulator::NeuroSimulator(const Config& config) 
    : config_(config), current_time_(0.0) {
    
    // Initialize core components
    BrainRouter::RoutingConfig router_config;
    router_config.autism_hypersensitivity = config_.autism_mode;
    router_config.ptsd_hypervigilance = config_.ptsd_overlay;
    router_config.amygdala_sensitivity = config_.ptsd_overlay ? 1.5 : 1.0;
    brain_router_ = std::make_unique<BrainRouter>(router_config);
    
    MultiModalFusion::FusionConfig fusion_config;
    fusion_config.autism_sensory_hypersensitivity = config_.autism_mode;
    fusion_config.ptsd_hypervigilance = config_.ptsd_overlay;
    multimodal_fusion_ = std::make_unique<MultiModalFusion>(fusion_config);
    
    // Initialize memory and flashback systems
    memory_overlay_ = std::make_unique<MemoryOverlay>();
    flashback_overlay_ = std::make_unique<FlashbackOverlay>();
    
    // Initialize brain regions
    initializeBrainRegions();
    
    if (config_.log_level == "DEBUG") {
        std::cout << "NeuroSimulator initialized with autism_mode=" << config_.autism_mode 
                  << ", ptsd_overlay=" << config_.ptsd_overlay << std::endl;
    }
}

NeuroSimulator::~NeuroSimulator() = default;

void NeuroSimulator::initializeBrainRegions() {
    // Configure base region settings
    BrainRegion::RegionConfig base_config;
    base_config.circuit_config.autism_mode = config_.autism_mode;
    base_config.circuit_config.ptsd_mode = config_.ptsd_overlay;
    base_config.circuit_config.ei_ratio = config_.excitation_ratio;
    base_config.circuit_config.inhibition_delay_ms = config_.inhibition_delay;
    
    // Initialize Amygdala
    base_config.region_name = "Amygdala";
    Amygdala::AmygdalaConfig amygdala_config;
    amygdala_config.autism_social_hypersensitivity = config_.autism_mode;
    amygdala_config.ptsd_hypervigilance = config_.ptsd_overlay;
    amygdala_config.ptsd_trauma_sensitivity = config_.ptsd_overlay ? 2.0 : 1.0;
    brain_regions_["Amygdala"] = std::make_unique<Amygdala>(base_config, amygdala_config);
    
    // Initialize other regions (simplified for now)
    base_config.region_name = "Hippocampus";
    brain_regions_["Hippocampus"] = std::make_unique<BrainRegion>(base_config);
    
    base_config.region_name = "Insula";
    brain_regions_["Insula"] = std::make_unique<BrainRegion>(base_config);
    
    base_config.region_name = "PFC";
    brain_regions_["PFC"] = std::make_unique<BrainRegion>(base_config);
    
    base_config.region_name = "Cerebellum";
    brain_regions_["Cerebellum"] = std::make_unique<BrainRegion>(base_config);
    
    base_config.region_name = "STG";
    brain_regions_["STG"] = std::make_unique<BrainRegion>(base_config);
    
    base_config.region_name = "ACC";
    brain_regions_["ACC"] = std::make_unique<BrainRegion>(base_config);
    
    // Register regions with brain router
    for (const auto& [name, region] : brain_regions_) {
        brain_router_->registerBrainRegion(name, region);
    }
}

NeuroSimulator::SimulationState NeuroSimulator::process(const MultiModalInput& input) {
    current_time_ += 1.0; // Increment simulation time
    
    SimulationState state;
    state.timestamp = current_time_;
    
    // Step 1: Multi-modal fusion
    MultiModalFusion::SensoryInput sensory_input;
    sensory_input.visual = input.visual_embedding;
    sensory_input.auditory = input.audio_embedding;
    sensory_input.vestibular = input.vestibular_embedding;
    sensory_input.interoceptive = input.interoceptive_embedding;
    sensory_input.timestamp = input.timestamp;
    
    auto fused_representation = multimodal_fusion_->fuse(sensory_input);
    
    // Step 2: Token analysis and brain routing
    std::vector<std::string> tokens;
    std::istringstream iss(input.text_tokens);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    auto region_activations = brain_router_->routeTokens(tokens, fused_representation.unified_embedding);
    
    // Step 3: Process activations in brain regions
    for (const auto& activation : region_activations) {
        if (brain_regions_.find(activation.region_name) != brain_regions_.end()) {
            double region_output = brain_regions_[activation.region_name]->processInput(
                activation.activation_strength, 1.0);
            state.region_activations[activation.region_name] = region_output;
        }
    }
    
    // Step 4: Check for flashback triggers (PTSD)
    if (config_.ptsd_overlay) {
        state.flashback_triggered = flashback_overlay_->checkTrigger(fused_representation.unified_embedding);
        if (state.flashback_triggered) {
            // Enhance amygdala activation during flashback
            if (state.region_activations.find("Amygdala") != state.region_activations.end()) {
                state.region_activations["Amygdala"] = std::min(1.0, 
                    state.region_activations["Amygdala"] * 1.5);
            }
        }
    }
    
    // Step 5: Update microcircuit state
    updateMicrocircuitState(state);
    
    // Step 6: Update multi-modal context
    state.multimodal_context.audio_pitch = fused_representation.fusion_metadata.dominant_modality == "auditory" ? "high" : "normal";
    state.multimodal_context.image_tag = fused_representation.fusion_metadata.dominant_modality == "visual" ? "detected" : "none";
    state.multimodal_context.body_state = fused_representation.autism_metrics.hypersensitivity_activation > 0.7 ? "rigid" : "neutral";
    state.multimodal_context.heartbeat = fused_representation.ptsd_metrics.threat_salience > 0.6 ? "elevated" : "normal";
    
    // Step 7: Generate response text (simplified)
    state.response_text = generateResponse(state);
    
    // Step 8: Store in memory
    memory_traces_.push_back(state);
    if (memory_traces_.size() > 1000) { // Limit memory size
        memory_traces_.erase(memory_traces_.begin());
    }
    
    // Step 9: Log state if debugging
    if (config_.log_level == "DEBUG") {
        logState(state);
    }
    
    return state;
}

NeuroSimulator::SimulationState NeuroSimulator::processText(const std::string& text) {
    MultiModalInput input;
    input.text_tokens = text;
    input.timestamp = current_time_;
    
    // Create minimal embeddings for text-only processing
    input.visual_embedding = Eigen::VectorXd::Zero(512);
    input.audio_embedding = Eigen::VectorXd::Zero(256);
    input.vestibular_embedding = Eigen::VectorXd::Zero(128);
    input.interoceptive_embedding = Eigen::VectorXd::Zero(64);
    
    return process(input);
}

void NeuroSimulator::updateMicrocircuitState(SimulationState& state) {
    // Calculate average excitation and inhibition across regions
    double total_excitation = 0.0;
    double total_inhibition = 0.0;
    int region_count = 0;
    
    for (const auto& [region_name, region] : brain_regions_) {
        const auto& microcircuit_state = region->getMicrocircuitState();
        total_excitation += microcircuit_state.excitatory_activity;
        total_inhibition += microcircuit_state.inhibitory_activity;
        region_count++;
    }
    
    if (region_count > 0) {
        state.microcircuit_state.excitation = total_excitation / region_count;
        state.microcircuit_state.inhibition = total_inhibition / region_count;
    }
    
    // Apply autism modifications
    if (config_.autism_mode) {
        state.microcircuit_state.excitation *= config_.excitation_ratio;
        state.microcircuit_state.inhibition *= 0.7; // Reduced inhibition
    }
    
    // Apply PTSD modifications
    if (config_.ptsd_overlay) {
        state.microcircuit_state.inhibition *= 0.8; // Delayed/reduced inhibition
    }
    
    // Detect looping (hyperexcitation)
    state.microcircuit_state.looping = (state.microcircuit_state.excitation / 
                                       std::max(0.1, state.microcircuit_state.inhibition)) > 2.0;
}

std::string NeuroSimulator::generateResponse(const SimulationState& state) {
    // Simple response generation based on brain state
    double amygdala_activation = state.region_activations.count("Amygdala") ? 
                                state.region_activations.at("Amygdala") : 0.0;
    
    if (state.flashback_triggered) {
        return "No. No. I don't want it.";
    } else if (amygdala_activation > 0.8) {
        return "I'm scared.";
    } else if (state.microcircuit_state.looping) {
        return "Too much. Too much.";
    } else if (config_.autism_mode && state.multimodal_context.body_state == "rigid") {
        return "Need quiet.";
    } else {
        return "Okay.";
    }
}

nlohmann::json NeuroSimulator::exportToJson(const SimulationState& state) const {
    nlohmann::json json_state;
    
    json_state["response"] = state.response_text;
    json_state["timestamp"] = state.timestamp;
    json_state["flashback_triggered"] = state.flashback_triggered;
    
    json_state["regions_triggered"] = nlohmann::json::object();
    for (const auto& [region, activation] : state.region_activations) {
        json_state["regions_triggered"][region] = activation;
    }
    
    json_state["microcircuit_state"]["excitation"] = state.microcircuit_state.excitation;
    json_state["microcircuit_state"]["inhibition"] = state.microcircuit_state.inhibition;
    json_state["microcircuit_state"]["looping"] = state.microcircuit_state.looping;
    
    json_state["multimodal_context"]["audio_pitch"] = state.multimodal_context.audio_pitch;
    json_state["multimodal_context"]["image_tag"] = state.multimodal_context.image_tag;
    json_state["multimodal_context"]["body_state"] = state.multimodal_context.body_state;
    json_state["multimodal_context"]["heartbeat"] = state.multimodal_context.heartbeat;
    
    return json_state;
}

void NeuroSimulator::updateConfig(const Config& config) {
    config_ = config;
    
    // Update component configurations
    if (brain_router_) {
        BrainRouter::RoutingConfig router_config = brain_router_->getConfig();
        router_config.autism_hypersensitivity = config_.autism_mode;
        router_config.ptsd_hypervigilance = config_.ptsd_overlay;
        brain_router_->updateConfig(router_config);
    }
    
    if (multimodal_fusion_) {
        MultiModalFusion::FusionConfig fusion_config = multimodal_fusion_->getConfig();
        fusion_config.autism_sensory_hypersensitivity = config_.autism_mode;
        fusion_config.ptsd_hypervigilance = config_.ptsd_overlay;
        multimodal_fusion_->updateConfig(fusion_config);
    }
    
    // Update brain region configurations
    for (const auto& [name, region] : brain_regions_) {
        // Update microcircuit configurations
        // This would require additional methods in BrainRegion class
    }
}

std::vector<NeuroSimulator::SimulationState> NeuroSimulator::getMemoryTraces() const {
    return memory_traces_;
}

void NeuroSimulator::clearMemory() {
    memory_traces_.clear();
    if (memory_overlay_) {
        memory_overlay_->clearMemory();
    }
}

void NeuroSimulator::addTraumaMemory(const Eigen::VectorXd& trauma_embedding, double trigger_threshold) {
    if (flashback_overlay_) {
        flashback_overlay_->addTraumaTemplate(trauma_embedding, trigger_threshold);
    }
    
    // Also add to amygdala if available
    auto amygdala_it = brain_regions_.find("Amygdala");
    if (amygdala_it != brain_regions_.end()) {
        auto* amygdala = dynamic_cast<Amygdala*>(amygdala_it->second.get());
        if (amygdala) {
            amygdala->addTraumaTemplate(trauma_embedding, trigger_threshold);
        }
    }
}

void NeuroSimulator::reset() {
    current_time_ = 0.0;
    memory_traces_.clear();
    
    // Reset all brain regions
    for (const auto& [name, region] : brain_regions_) {
        // This would require a reset method in BrainRegion
    }
    
    if (brain_router_) {
        brain_router_->clearHistory();
    }
    
    if (multimodal_fusion_) {
        multimodal_fusion_->clearHistory();
    }
}

void NeuroSimulator::logState(const SimulationState& state) const {
    std::cout << "[DEBUG] t=" << state.timestamp << " response=\"" << state.response_text << "\"" << std::endl;
    std::cout << "  Regions: ";
    for (const auto& [region, activation] : state.region_activations) {
        std::cout << region << "=" << activation << " ";
    }
    std::cout << std::endl;
    std::cout << "  E/I: " << state.microcircuit_state.excitation << "/"
              << state.microcircuit_state.inhibition;
    if (state.microcircuit_state.looping) std::cout << " [LOOPING]";
    if (state.flashback_triggered) std::cout << " [FLASHBACK]";
    std::cout << std::endl;
}

} // namespace neurosim
