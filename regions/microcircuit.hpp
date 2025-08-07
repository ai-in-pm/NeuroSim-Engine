#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Simulated neural microcircuit with GABA/Glutamate dynamics
 * 
 * This class models the fundamental excitatory/inhibitory balance in neural circuits
 * with specific considerations for:
 * - Autism: Elevated excitation/inhibition ratio, reduced inhibitory control
 * - PTSD: Delayed inhibition, memory flooding, hyperarousal
 * - Normal: Balanced E/I dynamics with proper gating
 */
class MicroCircuit {
public:
    /**
     * @brief Neurotransmitter state
     */
    struct NeurotransmitterState {
        double glutamate_level = 1.0;      ///< Excitatory neurotransmitter level
        double gaba_level = 1.0;           ///< Inhibitory neurotransmitter level
        double dopamine_level = 0.5;       ///< Reward/motivation modulation
        double serotonin_level = 0.5;      ///< Mood/anxiety modulation
        double norepinephrine_level = 0.5; ///< Arousal/attention modulation
        double acetylcholine_level = 0.5;  ///< Attention/learning modulation
    };

    /**
     * @brief Circuit configuration
     */
    struct CircuitConfig {
        double baseline_excitation = 1.0;     ///< Baseline excitatory drive
        double baseline_inhibition = 1.0;     ///< Baseline inhibitory drive
        double ei_ratio = 1.0;                ///< Excitation/Inhibition ratio
        double inhibition_delay_ms = 10.0;    ///< Inhibitory response delay
        double adaptation_rate = 0.1;         ///< Circuit adaptation rate
        double noise_level = 0.05;            ///< Neural noise level
        
        // Autism-specific parameters
        bool autism_mode = false;
        double autism_ei_elevation = 1.4;     ///< Elevated E/I ratio in autism
        double autism_inhibition_deficit = 0.7; ///< Reduced inhibitory control
        
        // PTSD-specific parameters
        bool ptsd_mode = false;
        double ptsd_inhibition_delay = 50.0;  ///< Delayed inhibition in PTSD
        double ptsd_hyperarousal = 1.5;       ///< Elevated baseline arousal
        double ptsd_memory_intrusion = 0.3;   ///< Memory intrusion probability
    };

    /**
     * @brief Circuit activation state
     */
    struct ActivationState {
        double excitatory_activity = 0.0;     ///< Current excitatory activity
        double inhibitory_activity = 0.0;     ///< Current inhibitory activity
        double net_activation = 0.0;          ///< Net circuit activation
        double firing_rate = 0.0;             ///< Simulated firing rate (Hz)
        
        bool in_oscillation = false;          ///< Whether circuit is oscillating
        double oscillation_frequency = 0.0;   ///< Oscillation frequency (Hz)
        bool hyperexcitable = false;          ///< Hyperexcitability state
        bool inhibition_failure = false;      ///< Inhibitory control failure
        
        NeurotransmitterState neurotransmitters; ///< Neurotransmitter levels
        
        // Temporal dynamics
        std::vector<double> activation_history; ///< Recent activation history
        double adaptation_level = 0.0;         ///< Current adaptation state
        double fatigue_level = 0.0;            ///< Neural fatigue level
    };

public:
    /**
     * @brief Constructor
     * @param config Circuit configuration
     */
    explicit MicroCircuit(const CircuitConfig& config = CircuitConfig{});

    /**
     * @brief Process input and update circuit state
     * @param input_strength Input activation strength
     * @param dt Time step in milliseconds
     * @return Updated activation state
     */
    ActivationState process(double input_strength, double dt = 1.0);

    /**
     * @brief Apply external modulation (e.g., from other brain regions)
     * @param modulation_type Type of modulation ("excitatory", "inhibitory", "neuromodulatory")
     * @param strength Modulation strength
     * @param duration Duration in milliseconds
     */
    void applyModulation(const std::string& modulation_type, double strength, double duration = 100.0);

    /**
     * @brief Simulate neurotransmitter release
     * @param neurotransmitter Type ("glutamate", "gaba", "dopamine", etc.)
     * @param amount Release amount
     */
    void releaseNeurotransmitter(const std::string& neurotransmitter, double amount);

    /**
     * @brief Get current circuit state
     * @return Current activation state
     */
    const ActivationState& getCurrentState() const { return current_state_; }

    /**
     * @brief Update circuit configuration
     * @param config New configuration
     */
    void updateConfig(const CircuitConfig& config);

    /**
     * @brief Get current configuration
     * @return Current circuit config
     */
    const CircuitConfig& getConfig() const { return config_; }

    /**
     * @brief Reset circuit to baseline state
     */
    void reset();

    /**
     * @brief Simulate autism-specific circuit modifications
     */
    void enableAutismMode();

    /**
     * @brief Simulate PTSD-specific circuit modifications
     */
    void enablePTSDMode();

    /**
     * @brief Get activation history for analysis
     * @return Vector of historical activation states
     */
    std::vector<ActivationState> getActivationHistory() const;

    /**
     * @brief Detect pathological patterns in circuit activity
     * @return Vector of detected pattern names
     */
    std::vector<std::string> detectPathologicalPatterns() const;

private:
    CircuitConfig config_;
    ActivationState current_state_;
    std::vector<ActivationState> activation_history_;
    
    // Temporal dynamics
    double current_time_;
    std::vector<std::pair<double, double>> pending_modulations_; // (end_time, strength)
    
    // Internal processing methods
    void updateExcitatoryActivity(double input_strength, double dt);
    void updateInhibitoryActivity(double dt);
    void updateNeurotransmitters(double dt);
    void applyAdaptation(double dt);
    void addNoise(double dt);
    
    // Autism-specific processing
    void applyAutismModifications();
    
    // PTSD-specific processing
    void applyPTSDModifications();
    
    // Oscillation detection and analysis
    void detectOscillations();
    double calculateOscillationFrequency() const;
    
    // Pathological pattern detection
    bool detectHyperexcitability() const;
    bool detectInhibitionFailure() const;
    bool detectSeizureActivity() const;
    bool detectMemoryIntrusion() const;
    
    // Utility methods
    double calculateFiringRate(double net_activation) const;
    double applyActivationFunction(double input) const;
    void updateActivationHistory();
    void pruneOldHistory();
    
    // Constants
    static constexpr double MAX_FIRING_RATE = 200.0; // Hz
    static constexpr double HISTORY_LENGTH = 1000.0; // ms
    static constexpr size_t MAX_HISTORY_SIZE = 1000;
};

/**
 * @brief Base class for all brain regions
 * 
 * Provides common interface and functionality for brain region models
 */
class BrainRegion {
public:
    /**
     * @brief Region-specific configuration
     */
    struct RegionConfig {
        std::string region_name;
        MicroCircuit::CircuitConfig circuit_config;
        double baseline_activation = 0.1;
        double activation_threshold = 0.5;
        double max_activation = 1.0;
        std::vector<std::string> connected_regions;
    };

    /**
     * @brief Constructor
     * @param config Region configuration
     */
    explicit BrainRegion(const RegionConfig& config);

    /**
     * @brief Virtual destructor
     */
    virtual ~BrainRegion() = default;

    /**
     * @brief Process region-specific input
     * @param input Input activation
     * @param dt Time step
     * @return Region activation level
     */
    virtual double processInput(double input, double dt = 1.0) = 0;

    /**
     * @brief Get region name
     * @return Region identifier
     */
    const std::string& getName() const { return config_.region_name; }

    /**
     * @brief Get current activation level
     * @return Current activation (0-1)
     */
    double getCurrentActivation() const { return current_activation_; }

    /**
     * @brief Get microcircuit state
     * @return Current microcircuit state
     */
    const MicroCircuit::ActivationState& getMicrocircuitState() const;

protected:
    RegionConfig config_;
    std::unique_ptr<MicroCircuit> microcircuit_;
    double current_activation_;
    double current_time_;
};

} // namespace neurosim
