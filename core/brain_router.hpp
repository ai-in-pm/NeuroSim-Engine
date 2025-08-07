#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

// Forward declaration
class BrainRegion;

/**
 * @brief Routes LLM token activations to specific brain regions
 * 
 * This class implements the core token-to-brain mapping logic that:
 * - Analyzes token semantic content and emotional valence
 * - Maps tokens to anatomically-inspired brain regions
 * - Applies weighted activation based on token importance
 * - Considers autism and PTSD-specific routing patterns
 */
class BrainRouter {
public:
    /**
     * @brief Token analysis result
     */
    struct TokenAnalysis {
        std::string token;                  ///< Original token
        double emotional_valence = 0.0;     ///< Emotional charge (-1 to 1)
        double arousal_level = 0.0;         ///< Arousal/activation level (0 to 1)
        double social_relevance = 0.0;      ///< Social interaction relevance (0 to 1)
        double threat_level = 0.0;          ///< Perceived threat level (0 to 1)
        double sensory_intensity = 0.0;     ///< Sensory processing load (0 to 1)
        std::vector<std::string> semantic_categories; ///< Semantic classifications
    };

    /**
     * @brief Brain region activation result
     */
    struct RegionActivation {
        std::string region_name;            ///< Brain region identifier
        double activation_strength = 0.0;   ///< Activation intensity (0 to 1)
        double latency_ms = 0.0;            ///< Activation latency in milliseconds
        std::vector<std::string> contributing_tokens; ///< Tokens that activated this region
        std::string activation_reason;      ///< Why this region was activated
    };

    /**
     * @brief Routing configuration
     */
    struct RoutingConfig {
        bool autism_hypersensitivity = false;  ///< Enhanced sensory routing in autism
        bool ptsd_hypervigilance = false;      ///< Enhanced threat detection in PTSD
        double amygdala_sensitivity = 1.0;     ///< Amygdala activation threshold
        double prefrontal_inhibition = 1.0;    ///< PFC inhibitory control strength
        double social_processing_bias = 1.0;   ///< Social brain network sensitivity
        double sensory_gating = 1.0;           ///< Sensory filtering strength
    };

public:
    /**
     * @brief Constructor
     * @param config Routing configuration
     */
    explicit BrainRouter(const RoutingConfig& config = RoutingConfig{});

    /**
     * @brief Route tokens to brain regions
     * @param tokens Input token sequence
     * @param multimodal_context Additional sensory context
     * @return Vector of region activations
     */
    std::vector<RegionActivation> routeTokens(
        const std::vector<std::string>& tokens,
        const Eigen::VectorXd& multimodal_context = Eigen::VectorXd()
    );

    /**
     * @brief Analyze individual token characteristics
     * @param token Input token
     * @return Token analysis result
     */
    TokenAnalysis analyzeToken(const std::string& token) const;

    /**
     * @brief Update routing configuration
     * @param config New configuration
     */
    void updateConfig(const RoutingConfig& config);

    /**
     * @brief Get current configuration
     * @return Current routing config
     */
    const RoutingConfig& getConfig() const { return config_; }

    /**
     * @brief Register a brain region for routing
     * @param region_name Region identifier
     * @param region Brain region instance
     */
    void registerBrainRegion(const std::string& region_name, std::shared_ptr<BrainRegion> region);

    /**
     * @brief Get activation history for analysis
     * @return Vector of historical activations
     */
    std::vector<std::vector<RegionActivation>> getActivationHistory() const;

    /**
     * @brief Clear activation history
     */
    void clearHistory();

private:
    RoutingConfig config_;
    std::unordered_map<std::string, std::shared_ptr<BrainRegion>> brain_regions_;
    std::vector<std::vector<RegionActivation>> activation_history_;

    // Token analysis methods
    double calculateEmotionalValence(const std::string& token) const;
    double calculateArousalLevel(const std::string& token) const;
    double calculateSocialRelevance(const std::string& token) const;
    double calculateThreatLevel(const std::string& token) const;
    double calculateSensoryIntensity(const std::string& token) const;
    std::vector<std::string> classifySemantics(const std::string& token) const;

    // Region-specific routing methods
    RegionActivation routeToAmygdala(const std::vector<TokenAnalysis>& tokens) const;
    RegionActivation routeToHippocampus(const std::vector<TokenAnalysis>& tokens) const;
    RegionActivation routeToInsula(const std::vector<TokenAnalysis>& tokens) const;
    RegionActivation routeToPrefrontal(const std::vector<TokenAnalysis>& tokens) const;
    RegionActivation routeToCerebellum(const std::vector<TokenAnalysis>& tokens) const;
    RegionActivation routeToSTG(const std::vector<TokenAnalysis>& tokens) const; // Superior Temporal Gyrus
    RegionActivation routeToACC(const std::vector<TokenAnalysis>& tokens) const; // Anterior Cingulate Cortex

    // Autism-specific routing modifications
    void applyAutismModifications(std::vector<RegionActivation>& activations) const;
    
    // PTSD-specific routing modifications
    void applyPTSDModifications(std::vector<RegionActivation>& activations) const;

    // Utility methods
    double calculateLatency(const std::string& region_name, double activation_strength) const;
    std::string generateActivationReason(const std::string& region_name, 
                                       const std::vector<TokenAnalysis>& contributing_tokens) const;

    // Static token classification data
    static const std::unordered_map<std::string, double> emotional_lexicon_;
    static const std::unordered_map<std::string, double> threat_lexicon_;
    static const std::unordered_map<std::string, double> social_lexicon_;
    static const std::unordered_map<std::string, std::vector<std::string>> semantic_categories_;
};

} // namespace neurosim
