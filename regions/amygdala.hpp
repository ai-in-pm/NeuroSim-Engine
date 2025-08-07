#pragma once

#include "microcircuit.hpp"
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Amygdala brain region model
 * 
 * The amygdala is critical for:
 * - Threat detection and fear processing
 * - Emotional memory formation
 * - Fight-or-flight response initiation
 * - Social threat assessment
 * 
 * In autism: Hyperactivation to social stimuli, difficulty with emotional regulation
 * In PTSD: Hypervigilance, trauma-related hyperactivation, memory intrusion triggers
 */
class Amygdala : public BrainRegion {
public:
    /**
     * @brief Amygdala-specific configuration
     */
    struct AmygdalaConfig {
        double threat_sensitivity = 0.7;        ///< Sensitivity to threat-related stimuli
        double social_threat_bias = 0.5;        ///< Bias toward social threat detection
        double memory_consolidation_rate = 0.3; ///< Rate of emotional memory formation
        double habituation_rate = 0.1;          ///< Rate of threat habituation
        
        // Autism-specific parameters
        bool autism_social_hypersensitivity = false;
        double autism_threat_generalization = 1.5; ///< Broader threat generalization
        double autism_emotional_dysregulation = 1.3; ///< Reduced emotional control
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance = false;
        double ptsd_trauma_sensitivity = 2.0;   ///< Enhanced trauma-related activation
        double ptsd_memory_intrusion_rate = 0.4; ///< Rate of intrusive memory activation
        std::vector<Eigen::VectorXd> trauma_templates; ///< Stored trauma patterns
    };

    /**
     * @brief Amygdala activation state
     */
    struct AmygdalaState {
        double threat_level = 0.0;              ///< Current perceived threat level
        double emotional_arousal = 0.0;         ///< Emotional arousal level
        double fear_response = 0.0;             ///< Fear response intensity
        double social_anxiety = 0.0;            ///< Social anxiety level
        
        bool fight_flight_active = false;       ///< Fight-or-flight response active
        bool memory_consolidation_active = false; ///< Emotional memory formation
        bool trauma_flashback_triggered = false; ///< PTSD flashback state
        
        std::vector<std::string> detected_threats; ///< Currently detected threats
        std::vector<std::string> active_memories;  ///< Currently active emotional memories
        
        // Temporal dynamics
        double habituation_level = 0.0;         ///< Current habituation to stimuli
        double sensitization_level = 0.0;       ///< Current sensitization level
    };

public:
    /**
     * @brief Constructor
     * @param region_config Base region configuration
     * @param amygdala_config Amygdala-specific configuration
     */
    Amygdala(const RegionConfig& region_config, const AmygdalaConfig& amygdala_config = AmygdalaConfig{});

    /**
     * @brief Process input with threat detection and emotional processing
     * @param input Input activation strength
     * @param dt Time step in milliseconds
     * @return Amygdala activation level
     */
    double processInput(double input, double dt = 1.0) override;

    /**
     * @brief Process multi-modal threat assessment
     * @param visual_input Visual threat cues
     * @param auditory_input Auditory threat cues
     * @param social_context Social context information
     * @param dt Time step
     * @return Threat assessment result
     */
    double processThreatAssessment(const Eigen::VectorXd& visual_input,
                                 const Eigen::VectorXd& auditory_input,
                                 const Eigen::VectorXd& social_context,
                                 double dt = 1.0);

    /**
     * @brief Process emotional memory consolidation
     * @param emotional_valence Emotional intensity (-1 to 1)
     * @param memory_content Memory content vector
     * @param dt Time step
     */
    void processMemoryConsolidation(double emotional_valence, 
                                  const Eigen::VectorXd& memory_content,
                                  double dt = 1.0);

    /**
     * @brief Check for trauma-related activation (PTSD)
     * @param input_pattern Current input pattern
     * @return Trauma match strength (0-1)
     */
    double checkTraumaActivation(const Eigen::VectorXd& input_pattern);

    /**
     * @brief Add trauma template for PTSD simulation
     * @param trauma_pattern Trauma-associated pattern
     * @param sensitivity Sensitivity threshold for activation
     */
    void addTraumaTemplate(const Eigen::VectorXd& trauma_pattern, double sensitivity = 0.8);

    /**
     * @brief Get current amygdala state
     * @return Current state
     */
    const AmygdalaState& getAmygdalaState() const { return amygdala_state_; }

    /**
     * @brief Update amygdala configuration
     * @param config New configuration
     */
    void updateConfig(const AmygdalaConfig& config);

    /**
     * @brief Simulate fear conditioning
     * @param conditioned_stimulus CS pattern
     * @param unconditioned_stimulus US intensity
     */
    void simulateFearConditioning(const Eigen::VectorXd& conditioned_stimulus, 
                                double unconditioned_stimulus);

    /**
     * @brief Simulate fear extinction
     * @param extinction_stimulus Extinction pattern
     * @param extinction_strength Extinction learning rate
     */
    void simulateFearExtinction(const Eigen::VectorXd& extinction_stimulus,
                              double extinction_strength = 0.1);

    /**
     * @brief Get emotional memory traces
     * @return Vector of stored emotional memories
     */
    std::vector<std::pair<Eigen::VectorXd, double>> getEmotionalMemories() const;

private:
    AmygdalaConfig amygdala_config_;
    AmygdalaState amygdala_state_;
    
    // Memory storage
    std::vector<std::pair<Eigen::VectorXd, double>> emotional_memories_; // (pattern, valence)
    std::vector<std::pair<Eigen::VectorXd, double>> fear_memories_;      // (CS, strength)
    
    // Internal processing methods
    double calculateThreatLevel(const Eigen::VectorXd& input) const;
    double calculateSocialThreat(const Eigen::VectorXd& social_context) const;
    double calculateEmotionalArousal(double threat_level, double input_strength) const;
    
    // Autism-specific processing
    void applyAutismModifications(double& activation, const Eigen::VectorXd& input);
    double calculateAutismSocialAnxiety(const Eigen::VectorXd& social_context) const;
    
    // PTSD-specific processing
    void applyPTSDModifications(double& activation, const Eigen::VectorXd& input);
    bool checkMemoryIntrusion(const Eigen::VectorXd& input) const;
    
    // Memory processing
    void updateEmotionalMemories(double emotional_valence, 
                               const Eigen::VectorXd& memory_content);
    double calculateMemoryMatch(const Eigen::VectorXd& input, 
                              const Eigen::VectorXd& stored_pattern) const;
    
    // Habituation and sensitization
    void updateHabituation(double input_strength, double dt);
    void updateSensitization(double threat_level, double dt);
    
    // Fear learning
    void updateFearConditioning(const Eigen::VectorXd& cs, double us_strength);
    void updateFearExtinction(const Eigen::VectorXd& extinction_stimulus, double strength);
    
    // Utility methods
    double applyHabituationEffect(double base_activation) const;
    double applySensitizationEffect(double base_activation) const;
    std::vector<std::string> identifyThreats(const Eigen::VectorXd& input) const;
    void updateActiveMemories(const Eigen::VectorXd& input);
};

} // namespace neurosim
