#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Multi-modal sensory integration and embedding fusion
 * 
 * This class implements probabilistic fusion of:
 * - Visual embeddings (CLIP-like visual features)
 * - Auditory embeddings (pitch, volume, spectral features)
 * - Vestibular embeddings (balance, motion, spatial orientation)
 * - Interoceptive embeddings (internal body state, arousal)
 * 
 * The fusion approximates how the brain integrates multiple sensory streams
 * with special considerations for autism (sensory hypersensitivity) and
 * PTSD (hypervigilance to threat-related sensory cues).
 */
class MultiModalFusion {
public:
    /**
     * @brief Configuration for multi-modal fusion
     */
    struct FusionConfig {
        double visual_weight = 0.4;         ///< Weight for visual modality
        double auditory_weight = 0.3;       ///< Weight for auditory modality
        double vestibular_weight = 0.15;    ///< Weight for vestibular modality
        double interoceptive_weight = 0.15; ///< Weight for interoceptive modality
        
        bool autism_sensory_hypersensitivity = false; ///< Enhanced sensory processing
        bool ptsd_hypervigilance = false;             ///< Enhanced threat detection
        
        double sensory_gating_threshold = 0.5;        ///< Threshold for sensory filtering
        double cross_modal_plasticity = 0.1;          ///< Cross-modal adaptation rate
        double temporal_integration_window = 500.0;   ///< Integration window in ms
    };

    /**
     * @brief Multi-modal sensory input
     */
    struct SensoryInput {
        Eigen::VectorXd visual;        ///< Visual feature vector
        Eigen::VectorXd auditory;      ///< Auditory feature vector
        Eigen::VectorXd vestibular;    ///< Vestibular feature vector
        Eigen::VectorXd interoceptive; ///< Interoceptive feature vector
        double timestamp = 0.0;        ///< Input timestamp
        double confidence = 1.0;       ///< Input confidence/quality
    };

    /**
     * @brief Fused multi-modal representation
     */
    struct FusedRepresentation {
        Eigen::VectorXd unified_embedding;           ///< Fused feature vector
        std::vector<double> modality_contributions;  ///< Per-modality contribution weights
        double fusion_confidence = 0.0;             ///< Confidence in fusion result
        double sensory_overload = 0.0;              ///< Sensory processing load (0-1)
        
        struct {
            std::string dominant_modality;           ///< Most influential sensory modality
            double cross_modal_conflict = 0.0;      ///< Conflict between modalities
            bool sensory_gating_active = false;     ///< Whether sensory gating occurred
        } fusion_metadata;
        
        // Autism-specific metrics
        struct {
            double hypersensitivity_activation = 0.0; ///< Sensory hypersensitivity level
            std::vector<std::string> overwhelming_modalities; ///< Modalities causing overload
        } autism_metrics;
        
        // PTSD-specific metrics
        struct {
            double threat_salience = 0.0;           ///< Threat-related sensory activation
            std::vector<std::string> trigger_modalities; ///< Modalities triggering hypervigilance
        } ptsd_metrics;
    };

public:
    /**
     * @brief Constructor
     * @param config Fusion configuration
     */
    explicit MultiModalFusion(const FusionConfig& config = FusionConfig{});

    /**
     * @brief Fuse multi-modal sensory inputs
     * @param input Sensory input data
     * @return Fused representation
     */
    FusedRepresentation fuse(const SensoryInput& input);

    /**
     * @brief Fuse multiple temporal inputs with integration window
     * @param inputs Vector of temporal sensory inputs
     * @return Temporally integrated fused representation
     */
    FusedRepresentation fuseTemporalSequence(const std::vector<SensoryInput>& inputs);

    /**
     * @brief Update fusion configuration
     * @param config New configuration
     */
    void updateConfig(const FusionConfig& config);

    /**
     * @brief Get current configuration
     * @return Current fusion config
     */
    const FusionConfig& getConfig() const { return config_; }

    /**
     * @brief Adapt fusion weights based on sensory history
     * @param sensory_history Recent sensory input history
     */
    void adaptWeights(const std::vector<SensoryInput>& sensory_history);

    /**
     * @brief Get fusion history for analysis
     * @return Vector of historical fusion results
     */
    std::vector<FusedRepresentation> getFusionHistory() const;

    /**
     * @brief Clear fusion history
     */
    void clearHistory();

    /**
     * @brief Simulate sensory overload conditions
     * @param overload_factor Intensity of sensory overload (0-2)
     * @return Modified fusion config for overload simulation
     */
    FusionConfig simulateSensoryOverload(double overload_factor) const;

private:
    FusionConfig config_;
    std::vector<FusedRepresentation> fusion_history_;
    std::vector<SensoryInput> temporal_buffer_;
    
    // Core fusion methods
    Eigen::VectorXd performWeightedFusion(const SensoryInput& input) const;
    std::vector<double> calculateModalityContributions(const SensoryInput& input) const;
    double calculateFusionConfidence(const SensoryInput& input, 
                                   const Eigen::VectorXd& fused_embedding) const;
    
    // Sensory processing methods
    double calculateSensoryOverload(const SensoryInput& input) const;
    std::string identifyDominantModality(const std::vector<double>& contributions) const;
    double calculateCrossModalConflict(const SensoryInput& input) const;
    bool applySensoryGating(const SensoryInput& input) const;
    
    // Autism-specific processing
    void applyAutismProcessing(FusedRepresentation& result, const SensoryInput& input) const;
    double calculateHypersensitivityActivation(const SensoryInput& input) const;
    std::vector<std::string> identifyOverwhelmingModalities(const SensoryInput& input) const;
    
    // PTSD-specific processing
    void applyPTSDProcessing(FusedRepresentation& result, const SensoryInput& input) const;
    double calculateThreatSalience(const SensoryInput& input) const;
    std::vector<std::string> identifyTriggerModalities(const SensoryInput& input) const;
    
    // Temporal integration
    void updateTemporalBuffer(const SensoryInput& input);
    Eigen::VectorXd performTemporalIntegration(const std::vector<SensoryInput>& inputs) const;
    
    // Utility methods
    double calculateModalityWeight(const std::string& modality, const SensoryInput& input) const;
    Eigen::VectorXd normalizeEmbedding(const Eigen::VectorXd& embedding) const;
    double calculateEmbeddingMagnitude(const Eigen::VectorXd& embedding) const;
};

} // namespace neurosim
