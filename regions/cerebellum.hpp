#pragma once

#include "microcircuit.hpp"
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Cerebellum brain region model
 * 
 * The cerebellum is critical for:
 * - Motor coordination and balance
 * - Cognitive coordination and timing
 * - Predictive processing and error correction
 * - Social cognition and language processing
 * 
 * In autism: Altered timing, coordination differences, social processing impacts
 * In PTSD: Hypervigilance affects coordination, startle responses
 */
class Cerebellum : public BrainRegion {
public:
    /**
     * @brief Cerebellum-specific configuration
     */
    struct CerebellumConfig {
        double motor_coordination_strength = 0.9; ///< Motor coordination capability
        double timing_precision = 0.8;           ///< Temporal processing precision
        double predictive_processing = 0.7;      ///< Predictive processing strength
        double error_correction_rate = 0.6;      ///< Error correction capability
        
        // Autism-specific parameters
        bool autism_timing_differences = false;
        double autism_coordination_variability = 1.2; ///< Increased coordination variability
        double autism_predictive_deficit = 0.7;       ///< Reduced predictive processing
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance_effects = false;
        double ptsd_startle_enhancement = 1.5;   ///< Enhanced startle responses
        double ptsd_coordination_disruption = 0.8; ///< Disrupted coordination
    };

    /**
     * @brief Constructor
     * @param region_config Base region configuration
     * @param cerebellum_config Cerebellum-specific configuration
     */
    Cerebellum(const RegionConfig& region_config, 
              const CerebellumConfig& cerebellum_config = CerebellumConfig{});

    /**
     * @brief Process input with coordination and timing
     * @param input Input activation strength
     * @param dt Time step in milliseconds
     * @return Cerebellum activation level
     */
    double processInput(double input, double dt = 1.0) override;

private:
    CerebellumConfig cerebellum_config_;
};

} // namespace neurosim
