#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

namespace neurosim {

// Forward declarations
class BrainRouter;
class MultiModalFusion;
class MemoryOverlay;
class FlashbackOverlay;
class BrainRegion;

/**
 * @brief Main NeuroSim Engine - simulates neurocognitive interactions
 * 
 * This class orchestrates the entire brain simulation, including:
 * - Token-to-brain-region routing
 * - Multi-modal sensory integration
 * - Memory formation and replay
 * - PTSD flashback overlays
 * - Autism-specific neural patterns
 */
class NeuroSimulator {
public:
    /**
     * @brief Configuration for the neural simulation
     */
    struct Config {
        bool autism_mode = false;           ///< Enable autism-specific neural patterns
        bool ptsd_overlay = false;          ///< Enable PTSD flashback mechanisms
        double excitation_ratio = 1.0;     ///< E/I ratio (elevated in autism)
        double inhibition_delay = 0.0;     ///< Inhibition delay (increased in PTSD)
        double memory_threshold = 0.7;     ///< Threshold for memory formation
        double flashback_sensitivity = 0.5; ///< Sensitivity to trauma triggers
        std::string log_level = "INFO";     ///< Logging verbosity
    };

    /**
     * @brief Simulation state for a single processing cycle
     */
    struct SimulationState {
        std::string response_text;                           ///< Generated LLM response
        std::unordered_map<std::string, double> region_activations; ///< Brain region activations
        struct {
            double excitation = 1.0;
            double inhibition = 1.0;
            bool looping = false;
        } microcircuit_state;                               ///< Neural microcircuit state
        
        struct {
            std::string audio_pitch = "normal";
            std::string image_tag = "none";
            std::string body_state = "neutral";
            std::string heartbeat = "normal";
        } multimodal_context;                               ///< Multi-modal sensory context
        
        double timestamp = 0.0;                             ///< Simulation timestamp
        bool flashback_triggered = false;                  ///< Whether flashback was triggered
        std::vector<std::string> active_memories;          ///< Currently active memory traces
    };

    /**
     * @brief Multi-modal input for the simulation
     */
    struct MultiModalInput {
        Eigen::VectorXd visual_embedding;      ///< Visual feature vector (CLIP-like)
        Eigen::VectorXd audio_embedding;       ///< Audio feature vector (pitch, volume, etc.)
        Eigen::VectorXd vestibular_embedding;  ///< Balance/motion vector
        Eigen::VectorXd interoceptive_embedding; ///< Internal body state vector
        std::string text_tokens;               ///< Input text tokens
        double timestamp = 0.0;                ///< Input timestamp
    };

public:
    /**
     * @brief Constructor
     * @param config Simulation configuration
     */
    explicit NeuroSimulator(const Config& config = Config{});
    
    /**
     * @brief Destructor
     */
    ~NeuroSimulator();

    /**
     * @brief Process a single simulation step
     * @param input Multi-modal input data
     * @return Current simulation state
     */
    SimulationState process(const MultiModalInput& input);

    /**
     * @brief Process text-only input (simplified interface)
     * @param text Input text
     * @return Current simulation state
     */
    SimulationState processText(const std::string& text);

    /**
     * @brief Get current configuration
     * @return Current config
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void updateConfig(const Config& config);

    /**
     * @brief Export simulation state to JSON
     * @param state Simulation state to export
     * @return JSON representation
     */
    nlohmann::json exportToJson(const SimulationState& state) const;

    /**
     * @brief Get longitudinal memory traces
     * @return Vector of stored memory states
     */
    std::vector<SimulationState> getMemoryTraces() const;

    /**
     * @brief Clear all memory traces
     */
    void clearMemory();

    /**
     * @brief Add trauma-encoded memory for PTSD simulation
     * @param trauma_embedding Embedding representing traumatic memory
     * @param trigger_threshold Sensitivity threshold for triggering
     */
    void addTraumaMemory(const Eigen::VectorXd& trauma_embedding, double trigger_threshold = 0.8);

    /**
     * @brief Reset simulation to initial state
     */
    void reset();

private:
    Config config_;
    
    // Core simulation components
    std::unique_ptr<BrainRouter> brain_router_;
    std::unique_ptr<MultiModalFusion> multimodal_fusion_;
    std::unique_ptr<MemoryOverlay> memory_overlay_;
    std::unique_ptr<FlashbackOverlay> flashback_overlay_;
    
    // Brain regions
    std::unordered_map<std::string, std::unique_ptr<BrainRegion>> brain_regions_;
    
    // Simulation state
    double current_time_;
    std::vector<SimulationState> memory_traces_;
    
    // Internal methods
    void initializeBrainRegions();
    void updateMicrocircuitState(SimulationState& state);
    void logState(const SimulationState& state) const;
};

} // namespace neurosim
