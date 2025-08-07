#pragma once

#include "microcircuit.hpp"
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Prefrontal Cortex brain region model
 * 
 * The PFC is critical for:
 * - Executive control and cognitive flexibility
 * - Working memory and attention regulation
 * - Inhibitory control and impulse regulation
 * - Social cognition and theory of mind
 * 
 * In autism: Executive function differences, cognitive rigidity
 * In PTSD: Impaired inhibitory control, hypervigilance, executive dysfunction
 */
class PrefrontalCortex : public BrainRegion {
public:
    /**
     * @brief PFC-specific configuration
     */
    struct PFCConfig {
        double executive_control_strength = 0.8; ///< Executive function capability
        double inhibitory_control_strength = 0.7; ///< Inhibitory control strength
        double working_memory_capacity = 0.6;    ///< Working memory capacity
        double cognitive_flexibility = 0.5;      ///< Cognitive flexibility
        
        // Autism-specific parameters
        bool autism_executive_differences = false;
        double autism_cognitive_rigidity = 1.4;  ///< Reduced cognitive flexibility
        double autism_inhibitory_deficit = 0.6;  ///< Reduced inhibitory control
        
        // PTSD-specific parameters
        bool ptsd_executive_dysfunction = false;
        double ptsd_inhibitory_impairment = 0.5; ///< Impaired inhibitory control
        double ptsd_hypervigilance_bias = 1.5;   ///< Attention bias to threats
    };

    /**
     * @brief Constructor
     * @param region_config Base region configuration
     * @param pfc_config PFC-specific configuration
     */
    PrefrontalCortex(const RegionConfig& region_config, 
                    const PFCConfig& pfc_config = PFCConfig{});

    /**
     * @brief Process input with executive control and inhibition
     * @param input Input activation strength
     * @param dt Time step in milliseconds
     * @return PFC activation level
     */
    double processInput(double input, double dt = 1.0) override;

private:
    PFCConfig pfc_config_;
};

} // namespace neurosim
