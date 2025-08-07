#pragma once

#include "microcircuit.hpp"
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Insula brain region model
 * 
 * The insula is critical for:
 * - Interoceptive awareness (body state monitoring)
 * - Emotional processing and empathy
 * - Pain and discomfort processing
 * - Social cognition and theory of mind
 * 
 * In autism: Altered interoceptive processing, social cognition differences
 * In PTSD: Hypervigilance to body states, emotional dysregulation
 */
class Insula : public BrainRegion {
public:
    /**
     * @brief Insula-specific configuration
     */
    struct InsulaConfig {
        double interoceptive_sensitivity = 0.7;  ///< Sensitivity to body signals
        double emotional_integration = 0.8;      ///< Emotional-body integration
        double social_cognition_strength = 0.6;  ///< Social processing capability
        double pain_sensitivity = 0.5;           ///< Pain/discomfort sensitivity
        
        // Autism-specific parameters
        bool autism_interoceptive_differences = false;
        double autism_social_cognition_reduction = 0.4; ///< Reduced social processing
        double autism_sensory_hypersensitivity = 1.4;   ///< Enhanced sensory sensitivity
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance = false;
        double ptsd_body_hyperawareness = 1.6;   ///< Enhanced body monitoring
        double ptsd_emotional_dysregulation = 1.3; ///< Emotional processing difficulties
    };

    /**
     * @brief Constructor
     * @param region_config Base region configuration
     * @param insula_config Insula-specific configuration
     */
    Insula(const RegionConfig& region_config, 
          const InsulaConfig& insula_config = InsulaConfig{});

    /**
     * @brief Process input with interoceptive and emotional processing
     * @param input Input activation strength
     * @param dt Time step in milliseconds
     * @return Insula activation level
     */
    double processInput(double input, double dt = 1.0) override;

private:
    InsulaConfig insula_config_;
};

} // namespace neurosim
