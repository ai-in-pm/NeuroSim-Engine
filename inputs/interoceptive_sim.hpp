#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Interoceptive simulation for internal body state monitoring
 * 
 * This class simulates interoceptive processing, converting internal body
 * signals into embeddings for neural simulation. It includes:
 * - Cardiovascular signals (heart rate, blood pressure)
 * - Respiratory signals (breathing rate, depth)
 * - Gastrointestinal signals (hunger, nausea)
 * - Thermoregulatory signals (temperature, sweating)
 * - Pain and discomfort signals
 * - Autonomic nervous system state
 * - Autism-specific interoceptive processing differences
 * - PTSD-specific hypervigilance to body signals
 */
class InteroceptiveSim {
public:
    /**
     * @brief Interoceptive processing configuration
     */
    struct InteroceptiveConfig {
        size_t embedding_dimension = 64;         ///< Output embedding size
        double sensitivity_threshold = 0.1;      ///< Signal detection threshold
        bool enable_cardiovascular = true;       ///< Enable heart/circulation signals
        bool enable_respiratory = true;          ///< Enable breathing signals
        bool enable_gastrointestinal = true;     ///< Enable digestive signals
        bool enable_thermoregulatory = true;     ///< Enable temperature signals
        bool enable_pain_processing = true;      ///< Enable pain/discomfort signals
        
        // Autism-specific parameters
        bool autism_interoceptive_differences = false; ///< Altered interoceptive processing
        double autism_signal_amplification = 1.2;      ///< Enhanced signal intensity
        double autism_signal_confusion = 1.1;          ///< Difficulty interpreting signals
        bool autism_alexithymia = false;               ///< Difficulty identifying emotions
        double autism_body_awareness_variance = 1.3;   ///< Variable body awareness
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance = false;        ///< Enhanced body monitoring
        double ptsd_arousal_sensitivity = 1.5;   ///< Enhanced arousal detection
        bool ptsd_dissociation_tendency = false; ///< Tendency to dissociate from body
        double ptsd_panic_threshold = 0.7;       ///< Threshold for panic response
        bool ptsd_somatic_symptoms = false;      ///< Physical PTSD symptoms
    };

    /**
     * @brief Interoceptive input data structure
     */
    struct InteroceptiveInput {
        // Cardiovascular signals
        double heart_rate = 70.0;                ///< Heart rate (BPM)
        double heart_rate_variability = 0.05;    ///< HRV measure
        double blood_pressure_systolic = 120.0;  ///< Systolic BP (mmHg)
        double blood_pressure_diastolic = 80.0;  ///< Diastolic BP (mmHg)
        
        // Respiratory signals
        double breathing_rate = 16.0;            ///< Breaths per minute
        double breathing_depth = 0.5;            ///< Breathing depth (0-1)
        double oxygen_saturation = 0.98;         ///< Blood oxygen saturation
        
        // Gastrointestinal signals
        double hunger_level = 0.3;               ///< Hunger intensity (0-1)
        double nausea_level = 0.0;               ///< Nausea intensity (0-1)
        double digestive_comfort = 0.8;          ///< Digestive comfort (0-1)
        
        // Thermoregulatory signals
        double core_temperature = 37.0;          ///< Core body temperature (°C)
        double skin_temperature = 32.0;          ///< Skin temperature (°C)
        double sweating_level = 0.1;             ///< Sweating intensity (0-1)
        
        // Pain and discomfort
        double pain_level = 0.0;                 ///< Overall pain intensity (0-1)
        double muscle_tension = 0.2;             ///< Muscle tension level (0-1)
        double fatigue_level = 0.3;              ///< Fatigue level (0-1)
        
        // Autonomic state
        double sympathetic_activation = 0.3;     ///< Sympathetic nervous system (0-1)
        double parasympathetic_activation = 0.7; ///< Parasympathetic nervous system (0-1)
        
        double timestamp = 0.0;                  ///< Input timestamp
        std::string context = "resting";         ///< Current context/activity
    };

    /**
     * @brief Interoceptive processing result
     */
    struct InteroceptiveEmbedding {
        Eigen::VectorXd feature_embedding;       ///< Main interoceptive feature vector
        
        // Processed signals
        double overall_arousal = 0.0;            ///< Overall physiological arousal
        double stress_level = 0.0;               ///< Estimated stress level
        double comfort_level = 0.8;              ///< Overall bodily comfort
        std::string autonomic_state;             ///< Autonomic nervous system state
        
        // Signal quality and awareness
        double interoceptive_accuracy = 0.8;     ///< Accuracy of body signal detection
        double body_awareness = 0.7;             ///< General body awareness level
        std::string dominant_signal_type;        ///< Most prominent signal type
        
        // Emotional and cognitive correlates
        std::string emotional_state;             ///< Inferred emotional state
        double anxiety_level = 0.0;              ///< Anxiety level from body signals
        bool panic_indicators = false;           ///< Panic attack indicators
        
        // Autism-specific metrics
        struct {
            double signal_amplification = 0.0;   ///< Signal intensity amplification
            double interpretation_difficulty = 0.0; ///< Difficulty interpreting signals
            bool alexithymia_indicators = false; ///< Difficulty identifying emotions
            double body_awareness_variability = 0.0; ///< Variability in body awareness
            std::vector<std::string> confusing_signals; ///< Signals that are confusing
        } autism_metrics;
        
        // PTSD-specific metrics
        struct {
            double hypervigilance_level = 0.0;   ///< Body hypervigilance level
            bool dissociation_indicators = false; ///< Dissociation from body
            double panic_risk = 0.0;             ///< Risk of panic response
            std::vector<std::string> somatic_symptoms; ///< Physical PTSD symptoms
            bool flashback_body_memories = false; ///< Body memory activation
        } ptsd_metrics;
        
        double processing_confidence = 1.0;       ///< Confidence in processing result
        double processing_time_ms = 0.0;          ///< Simulated processing time
    };

public:
    /**
     * @brief Constructor
     * @param config Interoceptive processing configuration
     */
    explicit InteroceptiveSim(const InteroceptiveConfig& config = InteroceptiveConfig{});

    /**
     * @brief Process interoceptive input and generate embedding
     * @param input Interoceptive input data
     * @return Interoceptive embedding result
     */
    InteroceptiveEmbedding processInteroceptiveInput(const InteroceptiveInput& input);

    /**
     * @brief Process simulated body state scenario
     * @param state_description Text description of internal body state
     * @return Simulated interoceptive embedding
     */
    InteroceptiveEmbedding processSimulatedBodyState(const std::string& state_description);

    /**
     * @brief Simulate specific physiological states
     * @param state_type Type of state ("stress", "calm", "exercise", etc.)
     * @param intensity State intensity (0-1)
     * @return Interoceptive embedding for the state
     */
    InteroceptiveEmbedding simulatePhysiologicalState(const std::string& state_type, 
                                                    double intensity);

    /**
     * @brief Update processing configuration
     * @param config New configuration
     */
    void updateConfig(const InteroceptiveConfig& config);

    /**
     * @brief Get current configuration
     * @return Current interoceptive config
     */
    const InteroceptiveConfig& getConfig() const { return config_; }

    /**
     * @brief Calibrate interoceptive system (set baseline)
     * @param baseline_input Baseline interoceptive state
     */
    void calibrateBaseline(const InteroceptiveInput& baseline_input);

    /**
     * @brief Simulate stress response
     * @param stressor_type Type of stressor
     * @param intensity Stress intensity (0-1)
     * @return Stress response embedding
     */
    InteroceptiveEmbedding simulateStressResponse(const std::string& stressor_type, 
                                                double intensity);

    /**
     * @brief Get processing history for analysis
     * @return Vector of recent interoceptive processing results
     */
    std::vector<InteroceptiveEmbedding> getProcessingHistory() const;

    /**
     * @brief Clear processing history
     */
    void clearHistory();

private:
    InteroceptiveConfig config_;
    InteroceptiveInput baseline_state_;
    std::vector<InteroceptiveEmbedding> processing_history_;
    
    // Core interoceptive processing methods
    Eigen::VectorXd extractInteroceptiveFeatures(const InteroceptiveInput& input);
    double calculateOverallArousal(const InteroceptiveInput& input);
    double calculateStressLevel(const InteroceptiveInput& input);
    double calculateComfortLevel(const InteroceptiveInput& input);
    std::string assessAutonomicState(const InteroceptiveInput& input);
    
    // Signal processing and awareness
    double calculateInteroceptiveAccuracy(const InteroceptiveInput& input);
    double calculateBodyAwareness(const InteroceptiveInput& input);
    std::string identifyDominantSignalType(const InteroceptiveInput& input);
    
    // Emotional and cognitive processing
    std::string inferEmotionalState(const InteroceptiveInput& input);
    double calculateAnxietyLevel(const InteroceptiveInput& input);
    bool detectPanicIndicators(const InteroceptiveInput& input);
    
    // Autism-specific processing
    void applyAutismInteroceptiveProcessing(InteroceptiveEmbedding& result, 
                                          const InteroceptiveInput& input);
    double calculateSignalAmplification(const InteroceptiveInput& input);
    double calculateInterpretationDifficulty(const InteroceptiveInput& input);
    bool checkAlexithymiaIndicators(const InteroceptiveInput& input);
    double calculateBodyAwarenessVariability(const InteroceptiveInput& input);
    std::vector<std::string> identifyConfusingSignals(const InteroceptiveInput& input);
    
    // PTSD-specific processing
    void applyPTSDInteroceptiveProcessing(InteroceptiveEmbedding& result, 
                                        const InteroceptiveInput& input);
    double calculateHypervigilanceLevel(const InteroceptiveInput& input);
    bool checkDissociationIndicators(const InteroceptiveInput& input);
    double calculatePanicRisk(const InteroceptiveInput& input);
    std::vector<std::string> identifySomaticSymptoms(const InteroceptiveInput& input);
    bool checkFlashbackBodyMemories(const InteroceptiveInput& input);
    
    // Utility methods
    double calculateProcessingTime(const InteroceptiveInput& input);
    double calculateProcessingConfidence(const InteroceptiveEmbedding& result);
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);
    double calculateSignalDeviation(double current_value, double baseline_value);
    
    // Simulated interoceptive processing
    InteroceptiveEmbedding simulateInteroceptiveProcessing(const std::string& description);
    InteroceptiveInput generatePhysiologicalState(const std::string& state_type, double intensity);
    std::vector<std::string> parseBodyStateDescription(const std::string& description);
    
    // Static data for state classification
    static const std::vector<std::string> physiological_states_;
    static const std::vector<std::string> autonomic_states_;
    static const std::vector<std::string> emotional_states_;
    static const std::vector<std::string> signal_types_;
};

} // namespace neurosim
