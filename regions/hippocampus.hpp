#pragma once

#include "microcircuit.hpp"
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Hippocampus brain region model
 * 
 * The hippocampus is critical for:
 * - Episodic memory formation and retrieval
 * - Spatial navigation and context processing
 * - Pattern separation and completion
 * - Temporal sequence processing
 * 
 * In autism: Enhanced detail memory, reduced contextual binding
 * In PTSD: Fragmented memory formation, context-dependent retrieval deficits
 */
class Hippocampus : public BrainRegion {
public:
    /**
     * @brief Hippocampus-specific configuration
     */
    struct HippocampusConfig {
        double memory_formation_rate = 0.3;     ///< Rate of new memory encoding
        double pattern_separation_strength = 0.7; ///< Pattern separation capability
        double pattern_completion_strength = 0.6; ///< Pattern completion capability
        double context_binding_strength = 0.8;   ///< Contextual association strength
        
        // Autism-specific parameters
        bool autism_detail_enhancement = false;
        double autism_context_reduction = 0.6;   ///< Reduced contextual processing
        double autism_pattern_rigidity = 1.3;    ///< Enhanced pattern rigidity
        
        // PTSD-specific parameters
        bool ptsd_fragmentation = false;
        double ptsd_context_deficit = 0.5;       ///< Impaired context processing
        double ptsd_memory_intrusion = 0.3;      ///< Intrusive memory formation
    };

    /**
     * @brief Constructor
     * @param region_config Base region configuration
     * @param hippocampus_config Hippocampus-specific configuration
     */
    Hippocampus(const RegionConfig& region_config, 
               const HippocampusConfig& hippocampus_config = HippocampusConfig{});

    /**
     * @brief Process input with memory formation and retrieval
     * @param input Input activation strength
     * @param dt Time step in milliseconds
     * @return Hippocampus activation level
     */
    double processInput(double input, double dt = 1.0) override;

private:
    HippocampusConfig hippocampus_config_;
};

} // namespace neurosim
