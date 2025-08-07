#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief PTSD flashback and trauma reactivation engine
 * 
 * This class simulates:
 * - Trauma-encoded memory patterns
 * - Flashback trigger detection
 * - Memory flooding and intrusion
 * - Hypervigilance states
 * - Dissociation responses
 * - Re-experiencing symptoms
 */
class FlashbackOverlay {
public:
    /**
     * @brief Trauma template for pattern matching
     */
    struct TraumaTemplate {
        Eigen::VectorXd pattern_embedding;      ///< Trauma-associated pattern
        double trigger_threshold = 0.8;         ///< Sensitivity for activation
        double emotional_intensity = 1.0;       ///< Emotional charge of trauma
        std::vector<std::string> sensory_markers; ///< Associated sensory cues
        std::vector<std::string> contextual_cues; ///< Environmental triggers
        
        double activation_frequency = 0.0;      ///< How often this template activates
        double last_activation = 0.0;           ///< Last activation timestamp
        bool is_primary_trauma = false;         ///< Whether this is the core trauma
        
        // Trauma characteristics
        std::string trauma_type;                ///< Type of trauma (combat, social, etc.)
        double fragmentation_level = 0.0;       ///< How fragmented the memory is
        double avoidance_strength = 0.0;        ///< Tendency to avoid related stimuli
    };

    /**
     * @brief Flashback state information
     */
    struct FlashbackState {
        bool flashback_active = false;          ///< Whether flashback is occurring
        double intensity = 0.0;                 ///< Flashback intensity (0-1)
        double duration_ms = 0.0;               ///< How long flashback has been active
        std::string trigger_type;               ///< What triggered the flashback
        
        std::vector<std::string> active_memories; ///< Memories being re-experienced
        std::vector<std::string> sensory_intrusions; ///< Intrusive sensory experiences
        
        bool dissociation_active = false;       ///< Whether dissociation is occurring
        double hypervigilance_level = 0.0;      ///< Current hypervigilance intensity
        bool memory_flooding = false;           ///< Whether memory flooding is occurring
        
        // Physiological simulation
        double simulated_heart_rate = 70.0;     ///< Simulated heart rate during flashback
        double stress_hormone_level = 0.0;      ///< Simulated stress response
        bool fight_flight_active = false;       ///< Fight-or-flight activation
    };

    /**
     * @brief Flashback configuration
     */
    struct FlashbackConfig {
        double base_trigger_sensitivity = 0.7;  ///< Base sensitivity to triggers
        double hypervigilance_threshold = 0.6;  ///< Threshold for hypervigilance
        double dissociation_threshold = 0.8;    ///< Threshold for dissociation
        double memory_flooding_threshold = 0.9; ///< Threshold for memory flooding
        
        double flashback_duration_base = 5000.0; ///< Base flashback duration (ms)
        double flashback_intensity_decay = 0.1;  ///< Rate of flashback intensity decay
        double hypervigilance_decay = 0.05;      ///< Rate of hypervigilance decay
        
        bool enable_dissociation = true;        ///< Whether to simulate dissociation
        bool enable_memory_flooding = true;     ///< Whether to simulate memory flooding
        double trauma_generalization = 0.3;     ///< How broadly trauma generalizes
        
        // Combat PTSD specific (for your background)
        bool combat_ptsd_mode = false;          ///< Enable combat-specific patterns
        double combat_hypervigilance = 1.5;     ///< Enhanced hypervigilance for combat
        std::vector<std::string> combat_triggers; ///< Combat-specific trigger words
    };

public:
    /**
     * @brief Constructor
     * @param config Flashback system configuration
     */
    explicit FlashbackOverlay(const FlashbackConfig& config = FlashbackConfig{});

    /**
     * @brief Check if current input triggers a flashback
     * @param input_pattern Current sensory/cognitive input
     * @return Whether flashback was triggered
     */
    bool checkTrigger(const Eigen::VectorXd& input_pattern);

    /**
     * @brief Process ongoing flashback state
     * @param dt Time step in milliseconds
     * @return Current flashback state
     */
    FlashbackState processFlashback(double dt = 1.0);

    /**
     * @brief Add trauma template for trigger detection
     * @param trauma_pattern Trauma-associated pattern
     * @param trigger_threshold Sensitivity threshold
     * @param trauma_type Type of trauma
     */
    void addTraumaTemplate(const Eigen::VectorXd& trauma_pattern,
                          double trigger_threshold = 0.8,
                          const std::string& trauma_type = "general");

    /**
     * @brief Add combat-specific trauma template
     * @param combat_scenario Combat scenario embedding
     * @param intensity Trauma intensity
     * @param location Combat location/context
     */
    void addCombatTrauma(const Eigen::VectorXd& combat_scenario,
                        double intensity = 1.0,
                        const std::string& location = "Fallujah");

    /**
     * @brief Simulate hypervigilance scanning
     * @param environmental_input Current environment
     * @return Threat assessment and hypervigilance level
     */
    std::pair<double, std::vector<std::string>> simulateHypervigilance(
        const Eigen::VectorXd& environmental_input);

    /**
     * @brief Get current flashback state
     * @return Current state
     */
    const FlashbackState& getCurrentState() const { return current_state_; }

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void updateConfig(const FlashbackConfig& config);

    /**
     * @brief Get all trauma templates
     * @return Vector of stored trauma patterns
     */
    const std::vector<TraumaTemplate>& getTraumaTemplates() const { return trauma_templates_; }

    /**
     * @brief Clear all trauma templates
     */
    void clearTraumaTemplates();

    /**
     * @brief Get flashback history for analysis
     * @return Vector of historical flashback episodes
     */
    std::vector<FlashbackState> getFlashbackHistory() const;

    /**
     * @brief Simulate grounding techniques (therapeutic intervention)
     * @param grounding_strength Effectiveness of grounding (0-1)
     */
    void applyGroundingTechnique(double grounding_strength = 0.7);

    /**
     * @brief Get trauma activation statistics
     * @return Statistics about trauma pattern activations
     */
    struct TraumaStats {
        size_t total_templates = 0;
        size_t recent_activations = 0;
        double average_trigger_sensitivity = 0.0;
        std::string most_frequent_trigger_type;
        double hypervigilance_time_percentage = 0.0;
    };
    TraumaStats getTraumaStats() const;

private:
    FlashbackConfig config_;
    std::vector<TraumaTemplate> trauma_templates_;
    FlashbackState current_state_;
    std::vector<FlashbackState> flashback_history_;
    
    double current_time_;
    double flashback_start_time_;
    double last_hypervigilance_scan_;
    
    // Core processing methods
    double calculateTriggerMatch(const Eigen::VectorXd& input, 
                               const TraumaTemplate& template) const;
    void initiateFlashback(const TraumaTemplate& triggered_template);
    void updateFlashbackIntensity(double dt);
    void updateHypervigilance(double dt);
    
    // Combat PTSD specific methods
    void applyCombatPTSDModifications();
    bool detectCombatTriggers(const Eigen::VectorXd& input) const;
    std::vector<std::string> identifyCombatCues(const Eigen::VectorXd& input) const;
    
    // Dissociation simulation
    void processDissociation(double trigger_intensity);
    bool shouldTriggerDissociation(double intensity) const;
    
    // Memory flooding simulation
    void processMemoryFlooding(const TraumaTemplate& template);
    std::vector<std::string> generateFloodingMemories(const TraumaTemplate& template) const;
    
    // Physiological response simulation
    void updatePhysiologicalResponse(double intensity, double dt);
    double calculateHeartRateResponse(double intensity) const;
    double calculateStressHormoneLevel(double intensity) const;
    
    // Utility methods
    void updateTraumaTemplateStats(TraumaTemplate& template);
    void pruneOldHistory();
    double calculateGeneralizationEffect(const Eigen::VectorXd& input) const;
    std::vector<std::string> extractSensoryMarkers(const Eigen::VectorXd& input) const;
    
    // Combat-specific trigger patterns (based on Operation Phantom Fury context)
    static const std::vector<std::string> combat_trigger_words_;
    static const std::vector<std::string> fallujah_contextual_cues_;
};

} // namespace neurosim
